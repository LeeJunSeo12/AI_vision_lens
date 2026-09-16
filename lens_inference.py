# -*- coding: utf-8 -*-
"""
============================================================
 Lens Defect Detection - 2 Stage Inference Pipeline
============================================================
 Stage 1 : EfficientNet-B0   → color / tint / empty 분류
 Stage 2 : ConvNeXt-Large 4ch → OK / NG 판정

 [성능] 2-stage 전체 (학습 미사용 평가, min4 = 96.9%)
   갈색 color : OK 96.9% / NG 96.9%
   검정 color : OK 97.6% / NG 98.3%

 [사용법]
   python lens_inference.py <이미지폴더경로>
   예) python lens_inference.py ./test_images
   → 결과가 inference_result.csv 로 저장됨

 [의존 패키지]
   torch, torchvision, timm, pillow, pandas, numpy, scikit-image
   (scikit-image 는 color 모델의 Otsu 이진화에 필요)
============================================================
"""

# ┌────────────────────────────────────────────────────────┐
# │  ★ 환경 설정 — 여기만 수정하세요                          │
# └────────────────────────────────────────────────────────┘
from pathlib import Path

# --- 가중치 경로 (3개) ---
MODEL1_PATH = Path("models/best_efficientnet_lens_3class.pth")   # Stage1: 분류기
COLOR_PATH  = Path("models/lens_pure_brown_best.pth")            # Stage2: color (★신규)
TINT_PATH   = Path("models/lens_ok_vs_tint_4ch_best.pth")        # Stage2: tint (기존)

# --- 판정 임계값 (p_ng > threshold 이면 NG) ---
COLOR_THRESHOLD = 0.25   # color (신규 모델 최적값)
TINT_THRESHOLD  = 0.45   # tint  (기존 유지)

# --- 결과 저장 파일 ---
OUTPUT_CSV = "inference_result.csv"

# ════════════════════════════════════════════════════════════
#  이하 수정 불필요
# ════════════════════════════════════════════════════════════
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from timm import create_model
from skimage.filters import threshold_otsu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL1_CLASSES = ["color", "tint", "empty"]   # Stage1 출력 순서 (idx0=color)
MODEL_NAME_4CH = "convnext_large_in22k"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 4ch 정규화 (color/tint 공통)
MEAN4 = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
STD4  = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(4, 1, 1)


# ──────────────────────────────────────────────
#  가중치 로드 유틸
# ──────────────────────────────────────────────
def load_sd(path):
    """체크포인트 형식 흡수 + DataParallel prefix 제거"""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


def adapt_to_4ch(m):
    """ConvNeXt stem 첫 conv 를 3채널 → 4채널 확장 (4번째 = RGB 평균)"""
    old = m.stem[0]
    new = nn.Conv2d(4, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:4] = old.weight.mean(1, keepdim=True)
        if old.bias is not None:
            new.bias = old.bias
    m.stem[0] = new
    return m


def build_4ch_model(weight_path):
    m = create_model(MODEL_NAME_4CH, pretrained=False, num_classes=2)
    m = adapt_to_4ch(m)
    m.load_state_dict(load_sd(weight_path), strict=True)
    return m.to(DEVICE).eval()


# ──────────────────────────────────────────────
#  전처리
# ──────────────────────────────────────────────
# Stage1 (EfficientNet) : 224 / 3ch / ImageNet 정규화
tfm_stage1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def make_4ch_color(img, sz=512):
    """color 전용 : 512 / Otsu 이진화 (신규 color 모델 학습과 일치)"""
    x = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
    ])(img)
    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    gn = gray.numpy().astype("float32")
    try:
        thr = float(threshold_otsu(gn))     # Otsu
    except Exception:
        thr = float(gn.mean())              # 단일값 이미지 등 예외 시 fallback
    binary = (gray > thr).float().unsqueeze(0)
    x4 = torch.cat([x, binary], 0)
    return (x4 - MEAN4) / STD4

def make_4ch_tint(img, sz=384):
    """tint 전용 : 384 / median 이진화 (기존 tint 모델 학습과 일치)"""
    x = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
    ])(img)
    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    binary = (gray > gray.median()).float().unsqueeze(0)
    x4 = torch.cat([x, binary], 0)
    return (x4 - MEAN4) / STD4


# ──────────────────────────────────────────────
#  모델 로드
# ──────────────────────────────────────────────
def load_models():
    # Stage1
    m1 = efficientnet_b0(weights=None)
    m1.classifier[1] = nn.Linear(m1.classifier[1].in_features, 3)
    m1.load_state_dict(load_sd(MODEL1_PATH), strict=True)
    m1 = m1.to(DEVICE).eval()
    # Stage2
    color_m = build_4ch_model(COLOR_PATH)
    tint_m  = build_4ch_model(TINT_PATH)
    return m1, color_m, tint_m


# ──────────────────────────────────────────────
#  추론 (1장)
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_one(img_path, m1, color_m, tint_m):
    """반환: dict(file, route, p_ng, pred, final_folder)"""
    img = Image.open(img_path).convert("RGB")

    # --- Stage 1 : 라우팅 ---
    logits = m1(tfm_stage1(img).unsqueeze(0).to(DEVICE))
    route = MODEL1_CLASSES[int(F.softmax(logits, 1)[0].argmax())]

    # --- Stage 2 : 판정 ---
    if route == "empty":
        # 빈 렌즈 슬롯 → NG 처리
        return {"file": img_path.name, "route": "empty",
                "p_ng": None, "pred": "NG", "final_folder": "Empty"}

    if route == "color":
        x = make_4ch_color(img).unsqueeze(0).to(DEVICE)    # 512 / Otsu
        p_ng = float(F.softmax(color_m(x), 1)[0, 0])       # idx0 = NG
        is_ng = p_ng > COLOR_THRESHOLD
        return {"file": img_path.name, "route": "color",
                "p_ng": round(p_ng, 4),
                "pred": "NG" if is_ng else "OK",
                "final_folder": "Color_NG" if is_ng else "Color_OK"}

    # route == "tint"
    x = make_4ch_tint(img).unsqueeze(0).to(DEVICE)         # 384 / median
    p_ng = float(F.softmax(tint_m(x), 1)[0, 0])            # idx0 = NG
    is_ng = p_ng > TINT_THRESHOLD
    return {"file": img_path.name, "route": "tint",
            "p_ng": round(p_ng, 4),
            "pred": "NG" if is_ng else "OK",
            "final_folder": "Tint_NG" if is_ng else "Tint_OK"}


def get_image_files(root):
    root = Path(root)
    return sorted([
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS
        and "__MACOSX" not in str(p)
        and not p.name.startswith("._")
        and ".ipynb_checkpoints" not in str(p)
    ])


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("사용법: python lens_inference.py <이미지폴더경로>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"[오류] 폴더를 찾을 수 없습니다: {target}")
        sys.exit(1)

    print(f"Device : {DEVICE}")
    print("모델 로드 중...")
    m1, color_m, tint_m = load_models()
    print(f"  Stage1 : {MODEL1_PATH.name}")
    print(f"  color  : {COLOR_PATH.name}  (512/Otsu, thr={COLOR_THRESHOLD})")
    print(f"  tint   : {TINT_PATH.name}  (384/median, thr={TINT_THRESHOLD})")

    imgs = get_image_files(target)
    print(f"\n추론 대상: {len(imgs)}장  ({target})")
    if not imgs:
        print("[오류] 이미지가 없습니다.")
        sys.exit(1)

    rows = []
    for i, ip in enumerate(imgs, 1):
        try:
            rows.append(predict_one(ip, m1, color_m, tint_m))
        except Exception as e:
            rows.append({"file": ip.name, "route": "ERROR",
                         "p_ng": None, "pred": "ERROR", "final_folder": str(e)[:50]})
        if i % 100 == 0:
            print(f"  {i}/{len(imgs)}")

    df = pd.DataFrame(rows)

    print("\n[라우팅 분포]")
    print(df["route"].value_counts().to_string())
    print("\n[OK/NG 판정]")
    print(df["pred"].value_counts().to_string())

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
