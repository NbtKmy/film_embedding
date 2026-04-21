"""
Qwen3-VL-Embedding helper — MLX backend for Apple Silicon.
Uses mlx-embeddings (https://github.com/Blaizzy/mlx-embeddings).
"""
import logging
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_embeddings import load

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nkamiy/Qwen3-VL-Embedding-8B-8bit-mlx"
DEFAULT_INSTRUCTION = "Represent the user's input."


class Qwen3VLEmbedder:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        default_instruction: str = DEFAULT_INSTRUCTION,
        **kwargs,
    ):
        logger.info(f"Loading MLX embedding model: {model_name_or_path}")
        self.model, self.processor = load(model_name_or_path, **kwargs)
        self.default_instruction = default_instruction

        # transformers >= 4.50 requires processor.image_ids to be set on the
        # inner Qwen3VLProcessor; mlx-embeddings bypasses ProcessorMixin.__init__
        # so this attribute may be missing.
        inner = getattr(self.processor, "processor", self.processor)
        if not hasattr(inner, "image_ids"):
            inner.image_ids = [getattr(inner, "image_token_id", None)]
        if not hasattr(inner, "video_ids"):
            inner.video_ids = [getattr(inner, "video_token_id", None)]
        if not hasattr(inner, "audio_ids"):
            inner.audio_ids = [None]

    def process(
        self,
        inputs: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of inputs. Each dict may contain:
          - "text": str
          - "image": str (file path) or PIL.Image
          - "video": list[str] (frame paths) — passed as multiple images
          - "instruction": str (optional override)

        Returns L2-normalized numpy array of shape (N, embedding_dim).
        """
        formatted = []
        for item in inputs:
            entry: Dict[str, Any] = {}

            text = item.get("text")
            if text:
                entry["text"] = text

            instruction = item.get("instruction", self.default_instruction)
            if instruction:
                entry["instruction"] = instruction

            # image: single path/PIL image
            image = item.get("image")
            if image:
                entry["image"] = image

            # video: list of frame paths → use first frame as representative image
            # mlx-embeddings Qwen3-VL accepts one image per entry
            video = item.get("video")
            if video and not image:
                if isinstance(video, list) and len(video) > 0:
                    mid = video[len(video) // 2]
                    entry["image"] = mid

            formatted.append(entry)

        embeddings = self.model.process(formatted, processor=self.processor)
        mx.eval(embeddings)

        arr = np.array(embeddings.astype(mx.float32))

        if normalize:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr = arr / norms

        return arr
