import asyncio
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from api.schemas import SentimentRequest, BatchSentimentRequest, SentimentResponse, BatchSentimentResponse
from api.model_loader import ModelLoader
from api.middleware import AuthMiddleware, LoggingMiddleware

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sentify.api")

model_loader = ModelLoader()
WARMUP_TEXT = "This is a warmup sentence for model initialization."


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load sklearn model (required)
    try:
        model_loader.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise e

    # Warmup sklearn
    try:
        model_loader.predict([WARMUP_TEXT], mode="fast")
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Model warmup failed (non-fatal): {e}")

    # Load DistilBERT in background — non-blocking
    asyncio.get_event_loop().run_in_executor(None, model_loader.load_transformer)

    yield


app = FastAPI(title="Sentify API", version="2.0.0", lifespan=lifespan)

app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware, model_loader=model_loader)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "model_type": model_loader.model_type,
        "transformer_ready": model_loader.transformer_ready,
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest, req: Request):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        start = time.perf_counter()
        results = model_loader.predict([request.text], mode=request.mode)
        inference_ms = round((time.perf_counter() - start) * 1000, 2)
        req.state.inference_time_ms = inference_ms

        label, confidence = results[0]
        used_mode = request.mode if (request.mode == "fast" or model_loader.transformer_ready) else "fast"
        return SentimentResponse(
            label=label,
            confidence=confidence,
            inference_time_ms=inference_ms,
            mode=used_mode,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict-batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest, req: Request):
    if not request.texts:
        raise HTTPException(status_code=400, detail="List of texts cannot be empty")

    try:
        start = time.perf_counter()
        results = model_loader.predict(request.texts, mode=request.mode)
        inference_ms = round((time.perf_counter() - start) * 1000, 2)
        req.state.inference_time_ms = inference_ms

        used_mode = request.mode if (request.mode == "fast" or model_loader.transformer_ready) else "fast"
        response_list = [
            SentimentResponse(label=label, confidence=conf, mode=used_mode)
            for label, conf in results
        ]
        return BatchSentimentResponse(
            predictions=response_list,
            inference_time_ms=inference_ms,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
