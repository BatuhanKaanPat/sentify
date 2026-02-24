from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SentimentRequest(BaseModel):
    text: str = Field(..., example="This movie was fantastic! I really enjoyed it.")
    mode: Literal["fast", "accurate"] = Field("fast", example="fast")


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., example=["I love this product.", "This is terrible."])
    mode: Literal["fast", "accurate"] = Field("fast", example="fast")


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    inference_time_ms: Optional[float] = None
    mode: Optional[str] = None


class BatchSentimentResponse(BaseModel):
    predictions: List[SentimentResponse]
    inference_time_ms: Optional[float] = None
