from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from skimage.filters import threshold_otsu

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torchvision import transforms
from torchvision.models import efficientnet_b0


"""
Lens Defect Detection - ver3 2 Stage Pipeline

Stage 1 : EfficientNet-B0 -> color / tint / empty 분류
Stage 2 : ConvNeXt-Large (4ch) -> OK / NG 판정

ver3 차이:
  - color 모델: lens_pure_brown_best.pth
  - color 전처리: 512 / Otsu 이진화
  - tint 전처리: 384 / median 이진화
  - threshold: color=0.25, tint=0.45
"""


# ============================================================
# 1. 경로 설정
# ============================================================
MODEL1_NAME = "best_efficientnet_lens_3class.pth"
COLOR_4CH_NAME = "lens_hardneg_best.pth"
TINT_4CH_NAME = "lens_tint_v3_best.pth"
TINT_V4_4CH_NAME = "lens_tint_dual_ep2_deploy.pth"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", Path.cwd() / "models"))


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHT_SEARCH_DIRS = [
    _model_dir(),
    Path.cwd(),
    Path.cwd() / "models",
    Path.cwd() / "weights",
    PROJECT_ROOT,
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "weights",
    Path("/mnt/data"),
]


# ============================================================
# 2. 모델 설정
# ============================================================
MODEL1_CLASSES = ["color", "tint", "empty"]

# 4ch 모델 출력 순서: index 0 = NG, index 1 = OK
COLOR_CLASSES = ["NG_color", "OK_color"]
TINT_CLASSES = ["NG_tint", "OK_tint"]

# NG 판정 임계값 (p_ng > threshold 이면 NG)
COLOR_THRESHOLD = 0.57
TINT_V3_THRESHOLD = 0.50
TINT_V4_THRESHOLD = 0.58
TINT_THRESHOLD = TINT_V3_THRESHOLD  # 하위 호환용

USE_TINT_ENSEMBLE = True

MODEL_NAME_4CH = "convnext_large_in22k"

MEAN4 = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
STD4 = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(4, 1, 1)

# Match the factory inference behavior for partially written/truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# 3. 유틸
# ============================================================
class InvalidImageError(ValueError):
    pass


def find_weight(filename: str) -> Path:
    """WEIGHT_SEARCH_DIRS 안에서 가중치 파일 탐색."""
    candidates = [Path(d) / filename for d in WEIGHT_SEARCH_DIRS]

    for path in candidates:
        if path.exists():
            print(f"weight found: {path}", flush=True)
            return path

    raise FileNotFoundError(
        f"가중치 못 찾음: {filename}\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def load_state_dict_safely(model: nn.Module, weight_path: Path, strict: bool = True) -> nn.Module:
    """체크포인트 형식 차이 흡수 + DataParallel prefix 제거."""
    ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=strict)
    return model


def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img.load()
        return img
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("유효한 이미지 파일이 아닙니다.") from exc


# ============================================================
# 4. Model_1 - EfficientNet-B0 (3-class)
# ============================================================
def build_model1() -> nn.Module:
    try:
        model = efficientnet_b0(weights=None)
    except TypeError:
        model = efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3)
    return model


model1_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_model1(model: nn.Module, img: Image.Image) -> tuple[str, np.ndarray]:
    x = model1_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    idx = int(np.argmax(probs))
    return MODEL1_CLASSES[idx], probs


# ============================================================
# 5. Model_2 - ConvNeXt-Large (4ch input)
# ============================================================
def adapt_model_to_4ch(model: nn.Module) -> nn.Module:
    """
    stem 첫 conv 를 3채널 -> 4채널로 확장.
    추가 채널 가중치는 기존 RGB 가중치 평균으로 초기화.
    """
    old_conv = model.stem[0]

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.stem[0] = new_conv
    return model


def build_4ch_model(weight_path: Path) -> nn.Module:
    model = create_model(MODEL_NAME_4CH, pretrained=False, num_classes=2)
    model = adapt_model_to_4ch(model)
    model = load_state_dict_safely(model, weight_path, strict=True)
    return model


def make_4ch_color(img: Image.Image, img_size: int = 512) -> torch.Tensor:
    """color 전용: 512 / Otsu 이진화."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    x = tfm(img)

    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    gray_np = gray.numpy().astype("float32")
    try:
        threshold = float(threshold_otsu(gray_np))
    except Exception:
        threshold = float(gray_np.mean())

    binary = (gray > threshold).float().unsqueeze(0)
    x4 = torch.cat([x, binary], dim=0)
    return (x4 - MEAN4) / STD4


def make_4ch_tint(img: Image.Image, img_size: int = 384) -> torch.Tensor:
    """tint 전용: 384 / grayscale median 이진화."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    x = tfm(img)

    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    binary = (gray > gray.median()).float().unsqueeze(0)
    x4 = torch.cat([x, binary], dim=0)
    return (x4 - MEAN4) / STD4


def predict_4ch(
    model: nn.Module,
    img: Image.Image,
    model_type: str,
) -> tuple[str, str, float, float, float]:
    """반환: (final_pred, final_folder, p_ng, p_ok, threshold)."""
    if model_type == "color":
        x = make_4ch_color(img).unsqueeze(0).to(DEVICE)
        threshold = COLOR_THRESHOLD
    elif model_type == "tint":
        x = make_4ch_tint(img).unsqueeze(0).to(DEVICE)
        threshold = TINT_THRESHOLD
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    p_ng = float(probs[0])
    p_ok = float(probs[1])

    if model_type == "color":
        final_pred = "NG_color" if p_ng >= threshold else "OK"
        final_folder = "Color_NG" if final_pred == "NG_color" else "Color_OK"
    else:
        final_pred = "NG_tint" if p_ng >= threshold else "OK"
        final_folder = "Tint_NG" if final_pred == "NG_tint" else "Tint_OK"

    return final_pred, final_folder, p_ng, p_ok, threshold


# ============================================================
# 6. 서버용 Pipeline 추론
# ============================================================
class LensDefectPipeline:
    def __init__(self):
        self.device = DEVICE
        self.model1_path = find_weight(MODEL1_NAME)
        self.color_4ch_path = find_weight(COLOR_4CH_NAME)
        self.tint_4ch_path = find_weight(TINT_4CH_NAME)

        self.model1 = build_model1()
        self.model1 = load_state_dict_safely(self.model1, self.model1_path, strict=True)
        self.model1 = self.model1.to(DEVICE).eval()

        self.color_model = build_4ch_model(self.color_4ch_path).to(DEVICE).eval()
        self.tint_model = build_4ch_model(self.tint_4ch_path).to(DEVICE).eval()

        self.tint_v4_model = None
        self.use_ensemble = False
        if USE_TINT_ENSEMBLE:
            try:
                self.tint_v4_path = find_weight(TINT_V4_4CH_NAME)
                self.tint_v4_model = build_4ch_model(self.tint_v4_path).to(DEVICE).eval()
                self.use_ensemble = True
                print(f"[틴트 앙상블 ON] v3={TINT_4CH_NAME}(thr {TINT_V3_THRESHOLD}) OR v4={TINT_V4_4CH_NAME}(thr {TINT_V4_THRESHOLD})", flush=True)
            except Exception as e:
                print(f"[경고] USE_TINT_ENSEMBLE=True 이나 v4 가중치가 없습니다 ({e}). 틴트는 v3 단독으로 동작합니다.", flush=True)

        self._lock = Lock()
        print(f"모든 모델 로드 완료. DEVICE: {DEVICE}", flush=True)

    def status(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "device": str(self.device),
            "model1_name": MODEL1_NAME,
            "color_4ch_name": COLOR_4CH_NAME,
            "tint_4ch_name": TINT_4CH_NAME,
            "tint_v4_4ch_name": TINT_V4_4CH_NAME if self.use_ensemble else None,
            "use_tint_ensemble": self.use_ensemble,
            "model_name_4ch": MODEL_NAME_4CH,
            "color_threshold": COLOR_THRESHOLD,
            "tint_v3_threshold": TINT_V3_THRESHOLD,
            "tint_v4_threshold": TINT_V4_THRESHOLD if self.use_ensemble else None,
            "tint_threshold": TINT_V3_THRESHOLD,
            "color_preprocess": "512/Otsu",
            "tint_preprocess": "384/median",
        }

    def predict_bytes(self, image_bytes: bytes, file_name: str | None = None) -> dict[str, Any]:
        img = read_image_from_bytes(image_bytes)
        return self.predict_image(img, file_name=file_name)

    def predict_image(self, img: Image.Image, file_name: str | None = None) -> dict[str, Any]:
        with self._lock:
            model1_pred, model1_probs = predict_model1(self.model1, img)

            result: dict[str, Any] = {
                "file_name": file_name,
                "model1_pred": model1_pred,
                "model1_prob_color": float(model1_probs[0]),
                "model1_prob_tint": float(model1_probs[1]),
                "model1_prob_empty": float(model1_probs[2]),
                "final_folder": None,
                "final_pred": None,
                "pred_okng": None,
                "p_ng": None,
                "p_ng_v3": None,
                "p_ng_v4": None,
                "p_ok": None,
                "threshold": None,
            }

            if model1_pred == "empty":
                result["final_folder"] = "Empty"
                result["final_pred"] = "empty"
                result["pred_okng"] = "NG"

            elif model1_pred == "color":
                final_pred, final_folder, p_ng, p_ok, threshold = predict_4ch(
                    self.color_model, img, model_type="color"
                )
                result["final_folder"] = final_folder
                result["final_pred"] = final_pred
                result["pred_okng"] = "NG" if final_folder == "Color_NG" else "OK"
                result["p_ng"] = p_ng
                result["p_ok"] = p_ok
                result["threshold"] = threshold

            elif model1_pred == "tint":
                # v3 추론
                final_pred, final_folder, p_ng_v3, p_ok_v3, threshold = predict_4ch(
                    self.tint_model, img, model_type="tint"
                )
                is_ng = p_ng_v3 >= TINT_V3_THRESHOLD
                p_ng_v4 = None

                # v4 앙상블 추론 (v3 OR v4)
                if self.use_ensemble and self.tint_v4_model is not None:
                    x_v4 = make_4ch_tint(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits_v4 = self.tint_v4_model(x_v4)
                        probs_v4 = F.softmax(logits_v4, dim=1)[0].detach().cpu().numpy()
                    p_ng_v4 = float(probs_v4[0])
                    is_ng = is_ng or (p_ng_v4 >= TINT_V4_THRESHOLD)

                final_pred = "NG_tint" if is_ng else "OK"
                final_folder = "Tint_NG" if final_pred == "NG_tint" else "Tint_OK"

                result["final_folder"] = final_folder
                result["final_pred"] = final_pred
                result["pred_okng"] = "NG" if is_ng else "OK"
                result["p_ng"] = p_ng_v3  # 하위호환 (v3 값)
                result["p_ng_v3"] = p_ng_v3
                result["p_ng_v4"] = p_ng_v4
                result["p_ok"] = p_ok_v3
                result["threshold"] = TINT_V3_THRESHOLD

            result["is_defect"] = result["pred_okng"] == "NG"
            result["result"] = "불량" if result["is_defect"] else "정상"
            return result
