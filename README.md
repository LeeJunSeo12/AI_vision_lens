# AI Vision Lens

> 렌즈 이미지를 자동으로 판정하는 GPU 기반 불량 검출 API

**AI Vision Lens**는 렌즈 이미지를 `color`, `tint`, `empty`로 먼저 분류한 뒤, 유형별 2차 모델로 최종 **정상(OK) / 불량(NG)** 을 판정하는 FastAPI 서비스입니다. HTTP API 호출과 폴더 감시 자동 처리 방식을 모두 지원해 생산 현장 자동화에 사용할 수 있습니다.

## 주요 기능

- **2단계 모델 파이프라인**: 1차 모델이 렌즈 유형을 분류하고, 유형별 전문 모델이 불량 여부를 판정합니다.
- **틴트 앙상블 판정**: tint v3·v4 모델의 결과를 OR 조건으로 결합해 불량 검출 성능을 보완합니다.
- **REST API 제공**: 단일 이미지, 여러 이미지 일괄 판정, 서비스 상태 확인을 지원합니다.
- **입력 폴더 자동 감시**: 새 이미지가 들어오면 자동 판정 후 날짜별 출력 폴더에 저장합니다.
- **NVIDIA GPU 지원**: Docker Compose를 통한 GPU 추론 환경을 제공합니다.

## 판정 흐름

```text
렌즈 이미지
    |
    v
1차 — EfficientNet-B0
    |-- color --> 2차 color 모델 (512 px + Otsu) ------> Color_OK / Color_NG
    |-- tint  --> 2차 tint 모델 (384 px + median) -----> Tint_OK / Tint_NG
    |-- empty ------------------------------------------> Empty (항상 NG)
```

`tint` 이미지의 경우 v3와 v4 모델을 각각 실행합니다. 두 모델 중 하나라도 설정된 불량 임계값에 도달하면 최종 결과는 `NG`입니다.

## 모델 파일

아래 네 개의 모델 파일은 프로젝트 루트 경로에 있어야 합니다. Docker 이미지를 빌드할 때 컨테이너 내부의 `/app/models`로 복사됩니다.

| 파일                                | 용도                                 |
| ----------------------------------- | ------------------------------------ |
| `best_efficientnet_lens_3class.pth` | 1차 분류: `color` / `tint` / `empty` |
| `lens_hardneg_best.pth`             | 2차 color 정상/불량 판정             |
| `lens_tint_v3_best.pth`             | 2차 tint v3 판정                     |
| `lens_tint_dual_ep2_deploy.pth`     | 2차 tint v4 앙상블 판정              |

> 모델 체크포인트의 용량이 큽니다. GitHub에 올릴 때는 일반 Git 대신 [Git LFS](https://git-lfs.com/)로 관리하는 것을 권장합니다.

## 사전 요구 사항

- Docker Desktop 및 Docker Compose v2
- NVIDIA GPU 및 NVIDIA 드라이버
- Docker Desktop GPU 지원 또는 NVIDIA Container Toolkit
- Windows PowerShell (아래 명령어 기준)

프로젝트의 PyTorch 의존성은 CUDA 12.8 휠을 사용합니다. CUDA 사용이 가능한 NVIDIA GPU 환경에서의 실행을 기준으로 구성되어 있습니다.

## 빠른 시작

1. 저장소를 내려받고, 위의 모델 파일 4개가 프로젝트 루트에 있는지 확인합니다.

2. [`docker-compose.yml`](docker-compose.yml)을 열어 입력·출력 폴더의 로컬 경로를 본인 환경에 맞게 수정합니다.

   ```yaml
   volumes:
     - "C:/path/to/input:/data/input"
     - "C:/path/to/output:/data/output"
   ```

3. 이미지를 빌드하고 API를 시작합니다.

   ```powershell
   docker compose up --build
   ```

4. 모델이 정상적으로 로드됐는지 확인합니다.

   ```powershell
   curl.exe http://localhost:8000/health
   ```

API 주소는 `http://localhost:8000`입니다. 브라우저에서 [`/docs`](http://localhost:8000/docs)에 접속하면 Swagger API 문서를 확인하고 요청을 직접 테스트할 수 있습니다.

## API 사용법

### `GET /health`

로드된 모델, 추론 장치(GPU/CPU), 판정 임계값, 폴더 감시 상태를 반환합니다.

```powershell
curl.exe http://localhost:8000/health
```

### `POST /predict`

이미지 한 장을 업로드해 판정합니다. JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF 형식을 지원합니다.

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "file=@C:\path\to\lens-image.jpg"
```

응답 예시:

```json
{
  "file_name": "lens-image.jpg",
  "model1_pred": "tint",
  "model1_prob_color": 0.01,
  "model1_prob_tint": 0.98,
  "model1_prob_empty": 0.01,
  "final_folder": "Tint_NG",
  "final_pred": "NG_tint",
  "pred_okng": "NG",
  "p_ng": 0.74,
  "p_ng_v3": 0.74,
  "p_ng_v4": 0.63,
  "p_ok": 0.26,
  "threshold": 0.5,
  "is_defect": true,
  "result": "불량"
}
```

### `POST /predict/batch`

이미지 여러 장을 한 번에 업로드해 판정합니다.

```powershell
curl.exe -X POST "http://localhost:8000/predict/batch" `
  -F "files=@C:\path\to\lens-01.jpg" `
  -F "files=@C:\path\to\lens-02.jpg"
```

## 입력 폴더 자동 감시

서비스는 기본적으로 `/data/input` 폴더를 1초마다 확인합니다. 새 이미지가 1초 이상 변경되지 않은 상태가 되면 판정하고, 결과를 마운트된 출력 폴더에 날짜별로 복사합니다.

```text
input/
  lens-001.jpg
        |
        v
output/
  2026-09-17/
    lens-001_NG.jpg
    processing_times.csv
```

`processing_times.csv`에는 정상 처리된 각 이미지의 파일명, 처리 시작·완료 시각, 총 처리 시간이 기록됩니다.

감시 동작의 기본값은 [`app/config.py`](app/config.py)에서 변경할 수 있습니다.

| 설정                              | 기본값  | 설명                                        |
| --------------------------------- | ------- | ------------------------------------------- |
| `WATCH_ENABLED`                   | `True`  | 폴더 자동 감시 사용 여부                    |
| `WATCH_POLL_INTERVAL_SECONDS`     | `1.0`   | 폴더 스캔 주기(초)                          |
| `WATCH_FILE_STABLE_SECONDS`       | `1.0`   | 이미지 처리 전 파일 안정화 대기 시간(초)    |
| `WATCH_PROCESS_EXISTING_ON_START` | `False` | 서버 시작 전부터 있던 입력 이미지 처리 여부 |

## 판정 임계값

| 경로    | 전처리                 | 불량(NG) 조건  |
| ------- | ---------------------- | -------------- |
| Color   | 512 px + Otsu 마스크   | `p_ng >= 0.57` |
| Tint v3 | 384 px + median 마스크 | `p_ng >= 0.50` |
| Tint v4 | 384 px + median 마스크 | `p_ng >= 0.58` |
| Empty   | —                      | 항상 `NG`      |

## 프로젝트 구조

```text
.
├── app/
│   ├── main.py                    # FastAPI 라우트 및 애플리케이션 수명 주기
│   ├── inference.py               # 모델 로드 및 2단계 추론
│   ├── watcher.py                 # 입력 폴더 감시 및 결과 파일 저장
│   ├── processing_time_logger.py  # 일자별 처리 시간 CSV 기록
│   └── config.py                  # 폴더 감시 설정
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── *.pth                          # 필수 모델 체크포인트 (Git LFS 권장)
```
