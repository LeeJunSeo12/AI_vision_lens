FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY best_efficientnet_lens_3class.pth ./models/best_efficientnet_lens_3class.pth
COPY lens_hardneg_best.pth ./models/lens_hardneg_best.pth
COPY lens_tint_v3_best.pth ./models/lens_tint_v3_best.pth
COPY lens_tint_dual_ep2_deploy.pth ./models/lens_tint_dual_ep2_deploy.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
