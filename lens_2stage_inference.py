#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lens_2stage_inference.py - 네오비전 렌즈 검사 2-stage 추론 (공장 배포/TTA 검증용, 단독 실행).

파이프라인
  Stage1  EfficientNet-B0            224x224 / ImageNet norm  → color / tint / empty
  Stage2A Color  ConvNeXt-L(4ch)     512x512 / RGB+Otsu       idx0=NG  thr=0.57
  Stage2B Tint   ConvNeXt-L(4ch)     384x384 / RGB+median     idx0=NG  thr=0.50
  empty                             → 즉시 NG (Stage2 없음)
  판정: p_ng >= threshold → NG.   p_ng = softmax[:,0].

전처리는 실제 학습/평가 코드와 1:1 동일:
  - Stage1  = step2_classify_batch.py
  - Color   = lens_factory_kit/common_train.py (512/Otsu/MEAN4·STD4)  ← 컬러 KPI 산출 코드
  - Tint    = tint_common.py (384/median/MEAN4·STD4)                  ← 틴트 v3 학습/평가 코드
  검증: 컬러 test 11,905장 @0.57(>=,정정114) → TP=1374/FP=27/FN=26/TN=10478 재현 (--selftest-color).

⚠️ 2-stage 전체 성능은 Stage1 라우팅 오분류(color↔tint↔empty)를 포함한다.
   즉 개별 Stage2 단독 KPI보다 낮을 수 있으며, 라우팅 오류가 최종 오판으로 전파된다.

사용 예
  단일 :  python lens_2stage_inference.py --input path/to/img.jpg
  폴더 :  python lens_2stage_inference.py --input path/to/folder            (하위폴더 재귀)
  대량 :  python lens_2stage_inference.py --input folder --batch 32         (선택 batch, 결과 동일)
  검증 :  python lens_2stage_inference.py --selftest-color

★ 평가 모드(검증관용) - 폴더 라벨(OK/NG)을 GT로 성능표+결과CSV 생성:
  python lens_2stage_inference.py --eval --data_dir 공장_전달데이터_최종 --output_dir results
  - 입력 구조 자동 인식: {Color,Tint}/test*/(유형/)?{OK,NG}/*.jpg
    (train/, excluded/, 숨김폴더는 자동 제외; 유형 하위폴더 brown/black/TYPE_00은 있으면 유형별 분해)
  - 출력: inference_results_YYYYMMDD_HHMMSS.csv (전 이미지 판정),
          mismatches.csv (오분류만), summary.txt (CM/지표/유형별/라우팅)
    → results/eval_YYYYMMDD_HHMMSS/ 아래 저장 (덮어쓰기 없음)

Windows(RTX 5080) 대응: 상대경로(__file__ 기준), workers 기본 0, __main__ 가드, utf-8,
콘솔 표는 ASCII 문자만 사용.
"""
import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.models import efficientnet_b0
from timm import create_model
from PIL import Image, ImageFile

# 학습/캐시와 동일하게 truncated 허용(전처리 일치). 완전 손상 파일은 open 단계에서 예외 → 개별 기록.
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from skimage.filters import threshold_otsu
    _HAS_OTSU = True
except Exception:
    _HAS_OTSU = False

# ── 경로/상수 (전부 __file__ 기준) ───────────────────────────────────────────
BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "weights"
STAGE1_W = WEIGHTS / "best_efficientnet_lens_3class.pth"
COLOR_W  = WEIGHTS / "lens_hardneg_best.pth"
TINT_W   = WEIGHTS / "lens_tint_v3_best.pth"        # 틴트 v3 (기존, 배포중)

# ══ threshold 상수 (한 곳에 모음) ═══════════════════════════════════════════════
COLOR_THRESHOLD   = 0.57       # Color Stage2 (변경 금지)
TINT_V3_THRESHOLD = 0.50       # Tint v3 (기존과 동일)
TINT_V4_THRESHOLD = 0.58       # Tint v4 (권장). 보수적으로 가려면 0.64
# 하위호환 별칭 (기존 코드/인자가 참조)
COLOR_THR = COLOR_THRESHOLD
TINT_THR  = TINT_V3_THRESHOLD

# ══ 틴트 앙상블(v3 OR v4) 설정 ═════════════════════════════════════════════════
#   최종 틴트 NG = (v3 p_ng >= TINT_V3_THRESHOLD) OR (v4 p_ng >= TINT_V4_THRESHOLD)
#   Stage1/Color 로직은 이 스위치와 무관하게 그대로 동작.
USE_TINT_ENSEMBLE = True        # False → 기존 v3 단독 동작(즉시 롤백)
TINT_V4_W = WEIGHTS / "lens_tint_dual_ep2_deploy.pth"   # 신규 v4 가중치(추가된 1개)

STAGE1_CLASSES = ["color", "tint", "empty"]      # argmax 0/1/2
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EVAL_B_S1, EVAL_B_COLOR, EVAL_B_TINT = 64, 8, 16   # 평가모드 내부 배치(결과는 배치와 무관)

MEAN4 = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
STD4  = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(4, 1, 1)
_STAGE1_TFM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ── 전처리 (학습/평가 코드와 1:1) ────────────────────────────────────────────
def pre_stage1(img):
    return _STAGE1_TFM(img)                                   # (3,224,224)

def pre_color(img):                                           # 512 / Otsu
    x = TF.to_tensor(TF.resize(img, (512, 512)))
    g = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    gn = g.numpy().astype("float32")
    try:
        thr = float(threshold_otsu(gn)) if _HAS_OTSU else float(gn.mean())
    except Exception:
        thr = float(gn.mean())
    b = (g > thr).float().unsqueeze(0)
    return (torch.cat([x, b], 0) - MEAN4) / STD4              # (4,512,512)

def pre_tint(img):                                            # 384 / median
    x = TF.to_tensor(TF.resize(img, (384, 384)))
    g = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]
    b = (g > g.median()).float().unsqueeze(0)
    return (torch.cat([x, b], 0) - MEAN4) / STD4              # (4,384,384)

# ── 모델 로드 (strict=True) ──────────────────────────────────────────────────
def _load_sd(p):
    sd = torch.load(str(p), map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:      sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}

def build_stage1(device):
    m = efficientnet_b0(num_classes=3)
    m.load_state_dict(_load_sd(STAGE1_W), strict=True)
    return m.to(device).eval()

def build_4ch(wp, device):
    m = create_model("convnext_large_in22k", pretrained=False, num_classes=2)
    old = m.stem[0]
    nc = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride,
                   old.padding, bias=(old.bias is not None))
    with torch.no_grad():
        nc.weight[:, :3] = old.weight
        nc.weight[:, 3] = old.weight.mean(dim=1)
        if old.bias is not None:
            nc.bias = old.bias
    m.stem[0] = nc
    m.load_state_dict(_load_sd(wp), strict=True)
    return m.to(device).eval()

class Models:
    def __init__(self, device, want_color=True, want_tint=True):
        self.device = device
        self.stage1 = build_stage1(device)
        self.color = build_4ch(COLOR_W, device) if want_color else None
        self.tint  = build_4ch(TINT_W, device) if want_tint else None   # 틴트 v3 (기존)
        # ── 틴트 v4 (앙상블용, 옵션·자동폴백) ──────────────────────────────────
        self.tint_v4 = None
        self.ensemble = False
        if want_tint and USE_TINT_ENSEMBLE:
            if TINT_V4_W.exists():
                self.tint_v4 = build_4ch(TINT_V4_W, device)   # v3와 동일 아키텍처(ConvNeXt-L 4ch,384), strict=True
                self.ensemble = True
            else:
                print(f"[경고] USE_TINT_ENSEMBLE=True 이나 v4 가중치가 없습니다: {TINT_V4_W}")
                print("       → 틴트는 v3 단독으로 동작합니다 (Stage1/Color 정상).")

# ── 추론 코어 (batch=1도 batch>1도 동일 경로 → 결과 동일 보장) ────────────────
def _amp_ctx(use_amp, device):
    if use_amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()

@torch.inference_mode()
def classify_chunk(paths, M, use_amp, color_thr=COLOR_THR, tint_thr=TINT_V3_THRESHOLD,
                   tint_v4_thr=TINT_V4_THRESHOLD):
    """paths(1~bs개) → 각 이미지 결과 dict 리스트. 손상 이미지는 개별 error 기록.
    틴트는 v3(OR v4 앙상블). p_ng=v3(하위호환), p_ng_v3/p_ng_v4 별도 기록."""
    device = M.device
    imgs, recs = [], []
    for p in paths:
        rec = dict(file=Path(p).name, path=str(p), route="", p_ng=None,
                   p_ng_v3=None, p_ng_v4=None, pred="", thr=None, error="")
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            imgs.append(None); rec["pred"] = "ERROR"; rec["error"] = f"open:{type(e).__name__}:{e}"
        recs.append(rec)

    valid = [i for i, im in enumerate(imgs) if im is not None]
    if not valid:
        return recs
    # Stage1
    try:
        x1 = torch.stack([pre_stage1(imgs[i]) for i in valid]).to(device)
        with _amp_ctx(use_amp, device):
            r = F.softmax(M.stage1(x1), 1).argmax(1).cpu().numpy()
        for j, i in enumerate(valid):
            recs[i]["route"] = STAGE1_CLASSES[int(r[j])]
    except Exception as e:
        for i in valid:
            recs[i]["pred"] = "ERROR"; recs[i]["error"] = f"stage1:{type(e).__name__}:{e}"
        return recs

    # Stage2 (route별 그룹 배치)
    def run_group(route, model, pre, thr):
        idxs = [i for i in valid if recs[i]["route"] == route and recs[i]["pred"] != "ERROR"]
        if not idxs:
            return
        if model is None:
            for i in idxs:
                recs[i]["pred"] = "ERROR"; recs[i]["error"] = f"{route}_model_미로드"
            return
        try:
            x = torch.stack([pre(imgs[i]) for i in idxs]).to(device)
            with _amp_ctx(use_amp, device):
                png = F.softmax(model(x), 1)[:, 0].float().cpu().numpy()
            for k, i in enumerate(idxs):
                pv = float(png[k])
                recs[i]["p_ng"] = pv; recs[i]["thr"] = thr
                recs[i]["pred"] = "NG" if pv >= thr else "OK"     # ★ p_ng >= threshold
        except Exception as e:
            for i in idxs:
                recs[i]["pred"] = "ERROR"; recs[i]["error"] = f"{route}:{type(e).__name__}:{e}"

    run_group("color", M.color, pre_color, color_thr)     # ★ Color 경로 무변경

    # ── 틴트 경로: v3 단독 또는 v3 OR v4 앙상블 ──────────────────────────────
    tidx = [i for i in valid if recs[i]["route"] == "tint" and recs[i]["pred"] != "ERROR"]
    if tidx:
        if M.tint is None:
            for i in tidx:
                recs[i]["pred"] = "ERROR"; recs[i]["error"] = "tint_model_미로드"
        else:
            try:
                xt = torch.stack([pre_tint(imgs[i]) for i in tidx]).to(device)
                with _amp_ctx(use_amp, device):
                    p3 = F.softmax(M.tint(xt), 1)[:, 0].float().cpu().numpy()
                    p4 = (F.softmax(M.tint_v4(xt), 1)[:, 0].float().cpu().numpy()
                          if M.tint_v4 is not None else None)
                for k, i in enumerate(tidx):
                    v3 = float(p3[k])
                    recs[i]["p_ng_v3"] = v3
                    recs[i]["p_ng"] = v3                       # 하위호환(기존 p_ng 컬럼=v3)
                    recs[i]["thr"] = tint_thr
                    ng = v3 >= tint_thr
                    if p4 is not None:                          # 앙상블: OR 결합
                        v4 = float(p4[k]); recs[i]["p_ng_v4"] = v4
                        ng = ng or (v4 >= tint_v4_thr)
                    recs[i]["pred"] = "NG" if ng else "OK"
            except Exception as e:
                for i in tidx:
                    recs[i]["pred"] = "ERROR"; recs[i]["error"] = f"tint:{type(e).__name__}:{e}"

    for i in valid:                       # empty → 즉시 NG
        if recs[i]["route"] == "empty" and recs[i]["pred"] != "ERROR":
            recs[i]["pred"] = "NG"; recs[i]["thr"] = None; recs[i]["p_ng"] = None
    return recs

def iter_chunks(seq, bs):
    for i in range(0, len(seq), bs):
        yield seq[i:i + bs]

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def gather_images(inp):
    p = Path(inp)
    if p.is_file():
        return [p] if p.suffix.lower() in IMG_EXTS else []
    return sorted([q for q in p.rglob("*")
                   if q.suffix.lower() in IMG_EXTS and not any(s.startswith(".") for s in q.parts)])

def fmt_png(v):
    return "" if v is None else f"{v:.8f}"      # p_ng 최소 8자리

def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 미검출 - 이 코드는 GPU 전용입니다(CPU 조용한 실행 금지). "
                           "NVIDIA 드라이버/CUDA 지원 torch를 확인하세요.")
    # 재현성: 같은 배치크기에서 run-to-run 결정적. (배치크기가 다르면 p_ng가 ~1e-6 수준
    #  달라질 수 있으나 판정 p_ng>=thr 결과는 불변. 공장 실시간은 batch=1 고정 권장.)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_env():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"cuDNN {torch.backends.cudnn.version()} | GPU {torch.cuda.get_device_name(0)}")

# ── 메인 추론 실행 (--input 모드, 기존과 동일) ───────────────────────────────
def run_inference(args):
    require_cuda()
    device = torch.device("cuda")
    print_env()
    print(f"[weights] stage1={STAGE1_W.name}  color={COLOR_W.name}  tint={TINT_W.name}")
    print(f"[thr] color={args.color_thr}  tint={args.tint_thr}  | 판정 p_ng>=thr | AMP={args.amp} | batch={args.batch}")
    print("[주의] 2-stage 결과는 Stage1 라우팅 오분류를 포함합니다(개별 Stage2 KPI와 다를 수 있음).")

    paths = gather_images(args.input)
    if not paths:
        print(f"[!] 이미지를 찾지 못했습니다: {args.input}"); return
    print(f"[입력] {len(paths):,}장  ({args.input})")

    t0 = time.perf_counter()
    M = Models(device)
    torch.cuda.synchronize()
    load_sec = time.perf_counter() - t0
    if M.ensemble:
        print(f"[틴트 앙상블 ON] v3={TINT_W.name}(thr {args.tint_thr}) OR v4={TINT_V4_W.name}(thr {args.tint_v4_thr})")
    else:
        print(f"[틴트 v3 단독] {TINT_W.name}(thr {args.tint_thr})  (앙상블 OFF 또는 v4 없음)")
    print(f"[모델 로드] {load_sec:.2f}s (이미지 처리시간과 분리 측정)")

    recs_all = []
    warm = max(0, args.warmup)
    done = 0
    for chunk in iter_chunks(paths, args.batch):
        torch.cuda.synchronize(); c0 = time.perf_counter()
        recs = classify_chunk(chunk, M, args.amp, args.color_thr, args.tint_thr, args.tint_v4_thr)
        torch.cuda.synchronize()
        each = (time.perf_counter() - c0) * 1000.0 / len(chunk)   # 이미지당 ms(배치는 균등분배)
        for r in recs:
            r["proc_ms"] = round(each, 3)
            r["warmup"] = done < warm                              # 앞 warmup장은 속도통계 제외
            recs_all.append(r); done += 1
    # steady-state 통계 (warmup 제외)
    steady = [r["proc_ms"] for r in recs_all if not r["warmup"] and r["pred"] != "ERROR"]
    n_err = sum(1 for r in recs_all if r["pred"] == "ERROR")

    # ── CSV 저장 (날짜시간 → 덮어쓰기 없음) ───────────────────────────────────
    outdir = Path(args.output_dir) if args.output_dir else (BASE / "inference_results")
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"lens_2stage_inference_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "route", "p_ng", "p_ng_v3", "p_ng_v4", "pred",
                    "threshold", "error", "proc_ms", "warmup", "path"])
        for r in recs_all:
            w.writerow([r["file"], r["route"], fmt_png(r["p_ng"]),
                        fmt_png(r.get("p_ng_v3")), fmt_png(r.get("p_ng_v4")), r["pred"],
                        ("" if r["thr"] is None else r["thr"]), r["error"],
                        f'{r["proc_ms"]:.3f}', int(r["warmup"]), r["path"]])

    # ── 요약 ──────────────────────────────────────────────────────────────────
    from collections import Counter
    routes = Counter(r["route"] for r in recs_all if r["pred"] != "ERROR")
    preds = Counter(r["pred"] for r in recs_all)
    print("\n" + "=" * 60)
    print(f"[결과] 총 {len(recs_all):,}장  | 라우팅 {dict(routes)}")
    print(f"       판정 {dict(preds)}  | 오류(손상 등) {n_err}장")
    if steady:
        arr = np.array(steady)
        spi = arr.mean() / 1000.0
        print(f"[속도] warmup {warm}장 제외 {len(steady):,}장 기준")
        print(f"       평균 {spi*1000:.2f} ms/장 = {spi:.5f} 초/장 = {spi/60:.6f} 분/매")
        print(f"       {1.0/spi:.1f} images/sec  |  1,000장 예상 {1000*spi/60:.2f}분")
    print(f"[CSV] {csv_path}")
    return csv_path

# ── 컬러 KPI 자체검증 (11,905 test, 캐시 경로 재추론) ─────────────────────────
def selftest_color(args):
    require_cuda(); device = torch.device("cuda"); print_env()
    cache = BASE / "lens_factory_kit/cache/ep3_test_png.npz"
    corr_p = BASE / "lens_factory_kit/label_corrections.json"
    if not cache.exists():
        print(f"[!] 캐시 없음: {cache}"); return
    import json
    z = np.load(cache, allow_pickle=True)
    paths = [Path(str(p)) for p in z["paths"]]
    ref_png = np.asarray(z["png"], dtype="float32")     # 원본(common_train) p_ng
    corr = {k: (1 if v.get("new", "OK") == "OK" else 0)
            for k, v in json.load(open(corr_p, encoding="utf-8")).items()}
    print(f"[selftest] 컬러 test {len(paths):,}장 재추론 (512/Otsu, AMP={args.amp})")

    color_m = build_4ch(COLOR_W, device)
    png = np.zeros(len(paths), "float32")
    for s in range(0, len(paths), args.batch):
        chunk = paths[s:s + args.batch]
        xs = []
        for p in chunk:
            xs.append(pre_color(Image.open(p).convert("RGB")))
        x = torch.stack(xs).to(device)
        with torch.inference_mode(), _amp_ctx(args.amp, device):
            png[s:s + len(chunk)] = F.softmax(color_m(x), 1)[:, 0].float().cpu().numpy()
        if s % 2560 == 0:
            print(f"    {s}/{len(paths)}", end="\r")

    def cat(p):
        pl = str(p).lower()
        for key, lab in [("/pure_brown/ng/", 0), ("/pure_brown/ok/", 1), ("/ng_black/", 0),
                         ("/ok_black/", 1), ("/type_00/ng", 0), ("/type_00/ok", 1)]:
            if key in pl:
                return lab
        return -1
    labels = []
    for p in paths:
        l = cat(p); nm = p.name
        if nm in corr:
            l = corr[nm]
        labels.append(l)
    labels = np.array(labels); v = labels >= 0
    def cm(scores):
        pn = scores[v] >= COLOR_THR; ng = labels[v] == 0
        return (int((pn & ng).sum()), int((pn & ~ng).sum()),
                int((~pn & ng).sum()), int((~pn & ~ng).sum()))
    TP, FP, FN, TN = cm(png)
    rTP, rFP, rFN, rTN = cm(ref_png)
    exp = (1374, 27, 26, 10478)
    print(f"\n  재추론 @0.57(>=): TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"  캐시     @0.57(>=): TP={rTP} FP={rFP} FN={rFN} TN={rTN}")
    print(f"  기대           : TP={exp[0]} FP={exp[1]} FN={exp[2]} TN={exp[3]}")
    max_diff = float(np.abs(png - ref_png).max())
    print(f"  p_ng 최대편차(재추론 vs 캐시) = {max_diff:.2e}")
    ok = (TP, FP, FN, TN) == exp
    print("  일치 - 전처리/index/threshold 동일" if ok else
          "  불일치 - AMP를 끄고(--no-amp) 재시도하거나 전처리 점검 필요")
    return ok

# ═════════════════════════════════════════════════════════════════════════════
# 평가 모드 (--eval) - 폴더 라벨(OK/NG)을 GT로 성능표 + 결과 CSV 생성 (검증관용)
#   로직은 배포 전 최종 검증(deploy_verify)에서 실측 재현이 확인된 것과 동일.
# ═════════════════════════════════════════════════════════════════════════════
def _discover(data_dir):
    """{Color,Tint}/test*/(유형/)?{OK,NG}/*.jpg 자동 인식.
    train/, excluded/, 숨김폴더 제외. 반환: {domain: [(path, gt, type), ...]}"""
    root = Path(data_dir)
    out = {"color": [], "tint": []}
    for dom_name, dom_key in (("Color", "color"), ("Tint", "tint")):
        dom = root / dom_name
        if not dom.is_dir():
            continue
        testdirs = sorted(d for d in dom.iterdir()
                          if d.is_dir() and d.name.lower().startswith("test"))
        for td in testdirs:
            for p in sorted(td.rglob("*")):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                parts = p.relative_to(td).parts
                if any(s.startswith(".") for s in parts) or "excluded" in [x.lower() for x in parts]:
                    continue
                labs = [x for x in parts[:-1] if x.upper() in ("OK", "NG")]
                if not labs:
                    continue                          # OK/NG 폴더 아래가 아니면 평가 제외
                gt = labs[-1].upper()
                typ = next((x for x in parts[:-1] if x.upper() not in ("OK", "NG")), "-")
                out[dom_key].append((p, gt, typ))
    # 루트 자체가 Color 또는 Tint일 때 (예: --data_dir .../Color)
    if not out["color"] and not out["tint"]:
        for dom_key in ("color", "tint"):
            if root.name.lower() == dom_key:
                sub = _discover(root.parent)
                return {dom_key: sub[dom_key], ("tint" if dom_key == "color" else "color"): []}
    return out

@torch.inference_mode()
def _batched_png(paths, model, pre, bs, tag):
    """Stage2 p_ng 일괄 (fp32, 결정적)."""
    out = np.zeros(len(paths), "float32")
    t0 = time.perf_counter()
    for s in range(0, len(paths), bs):
        chunk = paths[s:s + bs]
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in chunk]).cuda()
        out[s:s + len(chunk)] = F.softmax(model(x), 1)[:, 0].float().cpu().numpy()
        if (s // bs) % 40 == 0:
            print(f"    [{tag}] {s}/{len(paths)} ({time.perf_counter()-t0:.0f}s)", end="\r", flush=True)
    print(f"    [{tag}] {len(paths)}/{len(paths)} ({time.perf_counter()-t0:.0f}s)")
    return out

@torch.inference_mode()
def _batched_route(paths, stage1, bs):
    out = np.zeros(len(paths), "int64")
    t0 = time.perf_counter()
    for s in range(0, len(paths), bs):
        chunk = paths[s:s + bs]
        x = torch.stack([pre_stage1(Image.open(p).convert("RGB")) for p in chunk]).cuda()
        out[s:s + len(chunk)] = F.softmax(stage1(x), 1).argmax(1).cpu().numpy()
        if (s // bs) % 20 == 0:
            print(f"    [stage1] {s}/{len(paths)} ({time.perf_counter()-t0:.0f}s)", end="\r", flush=True)
    print(f"    [stage1] {len(paths)}/{len(paths)} ({time.perf_counter()-t0:.0f}s)")
    return np.array([STAGE1_CLASSES[int(k)] for k in out])

def _cm(gt_ng, pred_ng):
    TP = int((pred_ng & gt_ng).sum()); FP = int((pred_ng & ~gt_ng).sum())
    FN = int((~pred_ng & gt_ng).sum()); TN = int((~pred_ng & ~gt_ng).sum())
    return TP, FP, FN, TN

def _metrics(TP, FP, FN, TN):
    n = max(TP + FP + FN + TN, 1)
    acc = 100.0 * (TP + TN) / n
    p_ng = 100.0 * TP / max(TP + FP, 1); r_ng = 100.0 * TP / max(TP + FN, 1)
    f1_ng = 2 * p_ng * r_ng / max(p_ng + r_ng, 1e-9)
    p_ok = 100.0 * TN / max(TN + FN, 1); r_ok = 100.0 * TN / max(TN + FP, 1)
    f1_ok = 2 * p_ok * r_ok / max(p_ok + r_ok, 1e-9)
    return dict(n=n, acc=acc, p_ng=p_ng, r_ng=r_ng, f1_ng=f1_ng, p_ok=p_ok, r_ok=r_ok, f1_ok=f1_ok)

def _cm_block(title, TP, FP, FN, TN):
    m = _metrics(TP, FP, FN, TN)
    L = [f"[{title}]  (n={m['n']:,})",
         "  +-----------+-----------+-----------+",
         "  |           | actual NG | actual OK |",
         "  +-----------+-----------+-----------+",
         f"  | pred NG   | TP {TP:6,} | FP {FP:6,} |",
         f"  | pred OK   | FN {FN:6,} | TN {TN:6,} |",
         "  +-----------+-----------+-----------+",
         f"  Accuracy {m['acc']:6.2f}%",
         f"  NG기준: Precision {m['p_ng']:6.2f}%  Recall {m['r_ng']:6.2f}%  F1 {m['f1_ng']:6.2f}%",
         f"  OK기준: Precision {m['p_ok']:6.2f}%  Recall(통과율) {m['r_ok']:6.2f}%  F1 {m['f1_ok']:6.2f}%"]
    return "\n".join(L)

def eval_mode(args):
    require_cuda(); device = torch.device("cuda"); print_env()
    color_thr, tint_thr = args.color_thr, args.tint_thr
    print(f"[weights] stage1={STAGE1_W.name}  color={COLOR_W.name}  tint={TINT_W.name}")
    print(f"[thr] color={color_thr}  tint={tint_thr} | fp32 no-AMP(결정적) | 판정 p_ng>=thr")

    items = _discover(args.data_dir)
    nC, nT = len(items["color"]), len(items["tint"])
    if nC + nT == 0:
        print(f"[!] {args.data_dir} 에서 {{Color,Tint}}/test*/{{OK,NG}} 구조를 찾지 못했습니다."); return
    print(f"[데이터] Color {nC:,}장 / Tint {nT:,}장 (train·excluded·숨김폴더 자동 제외)")

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir) / f"eval_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    M = Models(device)

    # ── 1) Stage2 단독 (각 도메인 자기 모델). 틴트는 v3 OR v4 앙상블 ──
    def _tint_pred(paths, bs):
        """틴트 예측 boolean + (png_v3, png_v4). 앙상블 OFF면 v4=None."""
        p3 = _batched_png(paths, M.tint, pre_tint, bs, "tint-v3")
        if M.tint_v4 is not None:
            p4 = _batched_png(paths, M.tint_v4, pre_tint, bs, "tint-v4")
            pred = (p3 >= tint_thr) | (p4 >= args.tint_v4_thr)
        else:
            p4 = None; pred = p3 >= tint_thr
        return pred, p3, p4
    reports = []
    dom_data = {}
    if M.ensemble:
        reports.append(f"[틴트 앙상블] v3(thr {tint_thr}) OR v4(thr {args.tint_v4_thr})  (v4={TINT_V4_W.name})")
    for dom in ("color", "tint"):
        if not items[dom]:
            continue
        paths = [p for p, _, _ in items[dom]]
        gt = np.array([g == "NG" for _, g, _ in items[dom]])
        typ = np.array([t for _, _, t in items[dom]])
        print(f"[평가1] {dom} 단독 추론 {len(paths):,}장 ...")
        if dom == "color":
            png = _batched_png(paths, M.color, pre_color, EVAL_B_COLOR, "color")
            png4 = None; pred = png >= color_thr
            title = f"COLOR 단독 (thr {color_thr})"
        else:
            pred, png, png4 = _tint_pred(paths, EVAL_B_TINT)
            title = (f"TINT 단독 (v3 {tint_thr} OR v4 {args.tint_v4_thr})" if png4 is not None
                     else f"TINT 단독 (v3 {tint_thr})")
        dom_data[dom] = dict(paths=paths, gt=gt, typ=typ, png=png, png4=png4, pred=pred, thr=(color_thr if dom=="color" else tint_thr))
        TP, FP, FN, TN = _cm(gt, pred)
        reports.append(_cm_block(title, TP, FP, FN, TN))
        # 유형별 분해 (유형 하위폴더가 있을 때)
        utypes = [t for t in sorted(set(typ)) if t != "-"]
        for t in utypes:
            m = typ == t
            tTP, tFP, tFN, tTN = _cm(gt[m], pred[m])
            mm = _metrics(tTP, tFP, tFN, tTN)
            reports.append(f"  - 유형 {t:12s} n={mm['n']:6,}  TP={tTP} FP={tFP} FN={tFN} TN={tTN}"
                           f"  Acc {mm['acc']:.2f}%  P {mm['p_ng']:.2f}%  R {mm['r_ng']:.2f}%  F1 {mm['f1_ng']:.2f}%")

    # ── 2) Stage1 라우팅 + 통합 ──
    all_paths = [p for d in ("color", "tint") for p in dom_data.get(d, {}).get("paths", [])]
    all_gt = np.concatenate([dom_data[d]["gt"] for d in ("color", "tint") if d in dom_data])
    all_dom = np.array([d for d in ("color", "tint") for _ in dom_data.get(d, {}).get("paths", [])])
    print(f"[평가2] Stage1 라우팅 {len(all_paths):,}장 ...")
    route = _batched_route(all_paths, M.stage1, EVAL_B_S1)
    mis = {(a, b): int(((all_dom == a) & (route == b)).sum())
           for a in ("color", "tint") for b in ("color", "tint", "empty") if a != b}
    racc = 100.0 * float((route == all_dom).mean())
    reports.append(f"[Stage1 라우팅] 정확도 {racc:.3f}%  "
                   f"(color->tint {mis[('color','tint')]}, color->empty {mis[('color','empty')]}, "
                   f"tint->color {mis[('tint','color')]}, tint->empty {mis[('tint','empty')]})")

    # 통합 판정: 자기 라우팅은 단독(틴트 앙상블 포함) 결과 재사용, 오라우팅만 추가 추론
    integ_pred = np.zeros(len(all_paths), dtype=bool)
    integ_png  = np.full(len(all_paths), np.nan, "float32")   # v3/color p_ng (CSV)
    integ_png4 = np.full(len(all_paths), np.nan, "float32")   # v4 p_ng (틴트, CSV)
    off = {}; o = 0
    for d in ("color", "tint"):
        if d in dom_data:
            off[d] = o; o += len(dom_data[d]["paths"])
    for d in ("color", "tint"):
        if d not in dom_data:
            continue
        idx = np.arange(off[d], off[d] + len(dom_data[d]["paths"]))
        own = route[idx] == d
        integ_pred[idx[own]] = dom_data[d]["pred"][own]
        integ_png[idx[own]]  = dom_data[d]["png"][own]
        if d == "tint" and dom_data[d]["png4"] is not None:
            integ_png4[idx[own]] = dom_data[d]["png4"][own]
    # 오라우팅 → color
    needC = [i for i in range(len(all_paths)) if route[i] == "color" and all_dom[i] != "color"]
    if needC:
        pc = _batched_png([all_paths[i] for i in needC], M.color, pre_color, EVAL_B_COLOR, "cross-color")
        for k, i in enumerate(needC):
            integ_png[i] = pc[k]; integ_pred[i] = pc[k] >= color_thr
    # 오라우팅 → tint (앙상블)
    needT = [i for i in range(len(all_paths)) if route[i] == "tint" and all_dom[i] != "tint"]
    if needT:
        tp, t3, t4 = _tint_pred([all_paths[i] for i in needT], EVAL_B_TINT)
        for k, i in enumerate(needT):
            integ_png[i] = t3[k]; integ_pred[i] = bool(tp[k])
            if t4 is not None: integ_png4[i] = t4[k]
    pred_ng = np.where(route == "empty", True, integ_pred)
    TP, FP, FN, TN = _cm(all_gt, pred_ng)
    reports.append(_cm_block("2-STAGE 통합 (Stage1 라우팅 포함 실제 파이프라인)", TP, FP, FN, TN))

    # ── 3) 결과 CSV (전 이미지) + mismatches.csv ──
    all_typ = np.concatenate([dom_data[d]["typ"] for d in ("color", "tint") if d in dom_data])
    res_csv = outdir / f"inference_results_{stamp}.csv"
    mis_csv = outdir / "mismatches.csv"
    n_mis = 0
    with open(res_csv, "w", newline="", encoding="utf-8-sig") as f1, \
         open(mis_csv, "w", newline="", encoding="utf-8-sig") as f2:
        hdr = ["파일명", "경로", "렌즈종류(Stage1)", "유형(폴더)", "폴더GT", "최종판정",
               "p_ng(v3/color)", "p_ng_v4(tint)", "정답여부", "오분류유형"]
        w1 = csv.writer(f1); w1.writerow(hdr)
        w2 = csv.writer(f2); w2.writerow(hdr)
        for i, p in enumerate(all_paths):
            gt_s = "NG" if all_gt[i] else "OK"
            pd_s = "NG" if pred_ng[i] else "OK"
            case = ("TP" if (pred_ng[i] and all_gt[i]) else "FP" if pred_ng[i]
                    else "FN" if all_gt[i] else "TN")
            row = [Path(p).name, str(p), route[i], all_typ[i], gt_s, pd_s,
                   ("" if np.isnan(integ_png[i]) else f"{integ_png[i]:.8f}"),
                   ("" if np.isnan(integ_png4[i]) else f"{integ_png4[i]:.8f}"),
                   ("O" if gt_s == pd_s else "X"), case]
            w1.writerow(row)
            if gt_s != pd_s:
                w2.writerow(row); n_mis += 1
    reports.append(f"[불일치] 통합 기준 오분류 {n_mis}건 -> mismatches.csv")

    # ── 4) summary 저장 + 콘솔 ──
    head = ["=" * 66, "네오비전 렌즈 2-stage 평가 결과 (폴더 라벨 = GT, NG = 양성)",
            f"data_dir: {args.data_dir}", f"생성: {stamp} | thr color {color_thr} / tint {tint_thr} | fp32 no-AMP",
            f"평가 이미지: Color {nC:,} + Tint {nT:,} = {nC+nT:,}장 (excluded/train/숨김 제외)", "=" * 66]
    body = "\n\n".join(reports)
    (outdir / "summary.txt").write_text("\n".join(head) + "\n\n" + body + "\n", encoding="utf-8")
    print("\n" + "\n".join(head) + "\n\n" + body)
    print(f"\n[저장] {outdir}")
    print(f"  - inference_results_{stamp}.csv (전 {nC+nT:,}장 판정)")
    print(f"  - mismatches.csv ({n_mis}건)")
    print(f"  - summary.txt")
    return outdir

def main():
    ap = argparse.ArgumentParser(description="네오비전 렌즈 2-stage 추론(공장 배포/TTA 검증)")
    ap.add_argument("--input", help="이미지 파일 또는 폴더(재귀) - 단순 추론 모드")
    ap.add_argument("--eval", action="store_true", help="평가 모드: 폴더 라벨(OK/NG)=GT로 성능표+CSV")
    ap.add_argument("--data_dir", help="평가할 폴더 (예: 공장_전달데이터_최종)")
    ap.add_argument("--output_dir", default=str(BASE / "results"), help="결과 저장 위치(기본 results/)")
    ap.add_argument("--color_thr", type=float, default=COLOR_THR, help="Color NG threshold (기본 0.57)")
    ap.add_argument("--tint_thr", type=float, default=TINT_V3_THRESHOLD, help="Tint v3 NG threshold (기본 0.50)")
    ap.add_argument("--tint_v4_thr", type=float, default=TINT_V4_THRESHOLD, help="Tint v4 NG threshold (앙상블, 기본 0.58)")
    ap.add_argument("--batch", type=int, default=1, help="배치 크기(--input 모드; 공장 실시간=1 기본)")
    ap.add_argument("--warmup", type=int, default=3, help="GPU 워밍업 장수(속도통계 제외)")
    ap.add_argument("--amp", dest="amp", action="store_true", help="AMP(fp16) 사용(--input 모드)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="AMP 끔(기본, 결정적)")
    ap.set_defaults(amp=False)
    ap.add_argument("--selftest-color", action="store_true", help="컬러 KPI(11,905) 재현 검증")
    args = ap.parse_args()

    if args.selftest_color:
        selftest_color(args); return
    if args.eval:
        if not args.data_dir:
            ap.error("--eval 에는 --data_dir 이 필요합니다.")
        eval_mode(args); return
    if not args.input:
        ap.error("--input / --eval --data_dir / --selftest-color 중 하나를 지정하세요.")
    run_inference(args)

if __name__ == "__main__":   # Windows 필수
    main()
