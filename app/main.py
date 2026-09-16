from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import InvalidImageError, LensDefectPipeline
from app.watcher import ImageFolderWatcher


pipeline: LensDefectPipeline | None = None
folder_watcher: ImageFolderWatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, folder_watcher
    pipeline = LensDefectPipeline()
    folder_watcher = ImageFolderWatcher(pipeline)
    folder_watcher.start()
    try:
        yield
    finally:
        if folder_watcher is not None:
            folder_watcher.stop()


app = FastAPI(
    title="Lens Defect Detection API ver3",
    version="3.0.0",
    lifespan=lifespan,
)


def get_pipeline() -> LensDefectPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")
    return pipeline


@app.get("/health")
def health():
    status = get_pipeline().status()
    if folder_watcher is not None:
        status["folder_watcher"] = folder_watcher.status()
    return status


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")

    try:
        return get_pipeline().predict_bytes(image_bytes, file_name=file.filename)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="이미지 파일이 없습니다.")

    results = []
    pipe = get_pipeline()

    for file in files:
        image_bytes = await file.read()
        if not image_bytes:
            results.append({
                "file_name": file.filename,
                "error": "이미지 파일이 비어 있습니다.",
            })
            continue

        try:
            results.append(pipe.predict_bytes(image_bytes, file_name=file.filename))
        except InvalidImageError as exc:
            results.append({
                "file_name": file.filename,
                "error": str(exc),
            })

    return {"count": len(results), "results": results}
