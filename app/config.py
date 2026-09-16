from pathlib import Path


# Docker에서는 Windows 폴더를 docker-compose.yml의 volumes로 연결한 뒤,
# 여기에는 컨테이너 내부 경로를 지정합니다.
WATCH_ENABLED = True
WATCH_INPUT_DIR = Path("/data/input")
WATCH_OUTPUT_DIR = Path("/data/output")

# 새 파일 감시 주기와, 파일 복사가 끝났다고 판단하기 위한 대기 시간입니다.
WATCH_POLL_INTERVAL_SECONDS = 1.0
WATCH_FILE_STABLE_SECONDS = 1.0

# True면 서버 시작 전에 이미 입력 폴더에 있던 이미지도 처리합니다.
WATCH_PROCESS_EXISTING_ON_START = False
