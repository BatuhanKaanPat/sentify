import os
import sys
import pickle
import numpy as np
import logging
from typing import List, Tuple, Optional

sys.path.append(os.getcwd())

try:
    from src.preprocessing import preprocess_text, simple_tokenizer
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.preprocessing import preprocess_text, simple_tokenizer

logger = logging.getLogger("sentify.model_loader")

# Normalize any label format → human-readable
LABEL_MAP = {
    "0": "negative", "1": "positive",
    "neg": "negative", "pos": "positive",
    "negative": "negative", "positive": "positive",
    "NEGATIVE": "negative", "POSITIVE": "positive",
}


class ModelLoader:
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        self.model_path = model_path
        self.model = None          # sklearn pipeline
        self.model_type = None     # 'sklearn'
        self.transformer = None    # HuggingFace pipeline
        self.transformer_ready = False

    # ------------------------------------------------------------------ #
    #  Sklearn model                                                       #
    # ------------------------------------------------------------------ #
    def load_model(self):
        """Load sklearn pipeline from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        logger.info(f"Loading model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.model_type = 'sklearn'
        logger.info("Sklearn model loaded successfully.")

    # ------------------------------------------------------------------ #
    #  Transformer model (lazy, best-effort)                               #
    # ------------------------------------------------------------------ #
    def load_transformer(self):
        """
        Attempt to load DistilBERT sentiment pipeline.
        Logs a warning but does NOT crash if unavailable.
        """
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading DistilBERT model (this may take a moment)...")
            self.transformer = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,          # CPU
                truncation=True,
                max_length=512,
            )
            self.transformer_ready = True
            logger.info("DistilBERT loaded successfully.")
        except Exception as e:
            logger.warning(f"DistilBERT unavailable — accurate mode disabled. Reason: {e}")
            self.transformer_ready = False

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #
    def predict(
        self, texts: List[str], mode: str = "fast"
    ) -> List[Tuple[str, float]]:
        """
        Run inference.
        mode='fast'     → Logistic Regression  (ms latency)
        mode='accurate' → DistilBERT           (slower, context-aware)
        """
        if mode == "accurate":
            if not self.transformer_ready:
                logger.warning("Accurate mode requested but DistilBERT not loaded — falling back to fast.")
            else:
                return self._predict_transformer(texts)

        # Default / fallback: sklearn
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._predict_sklearn(texts)

    def _predict_sklearn(self, texts: List[str]) -> List[Tuple[str, float]]:
        try:
            labels = self.model.predict(texts)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(texts)
                confidences = np.max(probs, axis=1)
            else:
                confidences = [1.0] * len(texts)

            return [
                (LABEL_MAP.get(str(label).strip(), str(label)), float(conf))
                for label, conf in zip(labels, confidences)
            ]
        except Exception as e:
            logger.error(f"Sklearn inference failed: {e}")
            raise

    def _predict_transformer(self, texts: List[str]) -> List[Tuple[str, float]]:
        try:
            raw = self.transformer(texts, batch_size=8)
            return [
                (LABEL_MAP.get(r['label'], r['label']), round(r['score'], 4))
                for r in raw
            ]
        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")
            raise
