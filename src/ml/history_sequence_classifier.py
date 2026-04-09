"""History-sequence classifier for HPSketch-style command vectors.

This module provides a lightweight classifier that can consume `.h5` command
sequences (dataset key `vec`) and output a label/confidence signal for fusion.
The classifier supports:
1) rule/prototype scoring (no torch checkpoint needed), and
2) optional torch checkpoint inference when available.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ml.history_sequence_tools import (
    load_command_tokens_from_h5,
    sequence_statistics,
)

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:  # pragma: no cover - optional dependency
    import torch

    from src.ml.train.sequence_encoder import SequenceCommandClassifier

    HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    SequenceCommandClassifier = None


class HistorySequenceClassifier:
    """Predict labels from command history sequences."""

    def __init__(
        self,
        prototypes_path: Optional[str] = None,
        model_path: Optional[str] = None,
        min_sequence_length: int = 4,
        vec_key: str = "vec",
        command_col: int = 0,
        vocab_size: int = 512,
        prototype_token_weight: float = 1.0,
        prototype_bigram_weight: float = 1.0,
    ) -> None:
        self.prototypes_path = prototypes_path or os.getenv(
            "HISTORY_SEQUENCE_PROTOTYPES_PATH",
            "data/knowledge/history_sequence_prototypes_template.json",
        )
        self.model_path = model_path or os.getenv("HISTORY_SEQUENCE_MODEL_PATH", "")
        self.min_sequence_length = max(1, int(min_sequence_length))
        self.vec_key = str(vec_key)
        self.command_col = max(0, int(command_col))
        self.vocab_size = max(1, int(vocab_size))
        self.prototype_token_weight = self._resolve_non_negative_float(
            "HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT",
            float(prototype_token_weight),
        )
        self.prototype_bigram_weight = self._resolve_non_negative_float(
            "HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT",
            float(prototype_bigram_weight),
        )

        self.prototype_scores: Dict[str, Dict[int, float]] = {}
        self.prototype_bigram_scores: Dict[str, Dict[Tuple[int, int], float]] = {}
        self.prototype_bias: Dict[str, float] = {}
        self.model: Optional[Any] = None
        self.label_map: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}
        self.model_padding_idx: int = 0
        self.max_model_sequence_length: int = 512
        self._loaded_model = False

        self._load_prototypes()
        self._load_model()

    @staticmethod
    def _resolve_non_negative_float(env_key: str, default: float) -> float:
        raw = os.getenv(env_key)
        if raw is None:
            return max(0.0, float(default))
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("Invalid %s=%s, fallback to %.3f", env_key, raw, default)
            return max(0.0, float(default))

    def _load_prototypes(self) -> None:
        path = Path(self.prototypes_path).expanduser()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed loading history prototypes %s: %s", path, exc)
            return

        labels_payload = payload.get("labels", payload)
        if not isinstance(labels_payload, dict):
            return

        parsed_scores: Dict[str, Dict[int, float]] = {}
        parsed_bigram_scores: Dict[str, Dict[Tuple[int, int], float]] = {}
        parsed_bias: Dict[str, float] = {}
        for label, spec in labels_payload.items():
            label_text = str(label or "").strip()
            if not label_text or not isinstance(spec, dict):
                continue
            raw_weights = spec.get("token_weights", spec.get("tokens", {}))
            if not isinstance(raw_weights, dict):
                raw_weights = {}
            weights: Dict[int, float] = {}
            for token_key, weight in raw_weights.items():
                try:
                    token_id = int(token_key)
                    weight_f = float(weight)
                except Exception:
                    continue
                if token_id < 0:
                    continue
                weights[token_id] = weight_f
            if not weights:
                continue
            parsed_scores[label_text] = weights
            raw_bigram_weights = spec.get("bigram_weights", {})
            bigram_weights: Dict[Tuple[int, int], float] = {}
            if isinstance(raw_bigram_weights, dict):
                for pair_key, weight in raw_bigram_weights.items():
                    try:
                        left_raw, right_raw = str(pair_key).split(",", 1)
                        left = int(left_raw)
                        right = int(right_raw)
                        weight_f = float(weight)
                    except Exception:
                        continue
                    if left < 0 or right < 0:
                        continue
                    bigram_weights[(left, right)] = weight_f
            if not weights and not bigram_weights:
                continue
            parsed_bigram_scores[label_text] = bigram_weights
            parsed_bias[label_text] = float(spec.get("bias", 0.0) or 0.0)

        self.prototype_scores = parsed_scores
        self.prototype_bigram_scores = parsed_bigram_scores
        self.prototype_bias = parsed_bias

    def _load_model(self) -> None:
        if not HAS_TORCH or not self.model_path:
            return
        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            return
        try:
            checkpoint = torch.load(str(model_path), map_location="cpu")
            raw_label_map = checkpoint.get("label_map", {})
            if not isinstance(raw_label_map, dict) or not raw_label_map:
                raise ValueError("missing label_map in checkpoint")
            label_map: Dict[str, int] = {}
            for key, value in raw_label_map.items():
                try:
                    label_map[str(key)] = int(value)
                except Exception:
                    continue
            if not label_map:
                raise ValueError("invalid label_map in checkpoint")

            model_cfg = checkpoint.get("model_config", {})
            vocab_size = int(model_cfg.get("vocab_size", self.vocab_size))
            embedding_dim = int(model_cfg.get("embedding_dim", 64))
            hidden_dim = int(model_cfg.get("hidden_dim", 128))
            num_layers = int(model_cfg.get("num_layers", 1))
            dropout = float(model_cfg.get("dropout", 0.1))
            bidirectional = bool(model_cfg.get("bidirectional", False))
            padding_idx = int(model_cfg.get("padding_idx", 0))
            num_classes = max(1, max(label_map.values()) + 1)
            self.vocab_size = max(1, vocab_size)
            self.model_padding_idx = max(0, padding_idx)
            self.max_model_sequence_length = max(
                1,
                int(
                    model_cfg.get("max_sequence_length", self.max_model_sequence_length)
                ),
            )

            self.model = SequenceCommandClassifier(
                vocab_size=self.vocab_size,
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                padding_idx=self.model_padding_idx,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.label_map = label_map
            self.index_to_label = {idx: label for label, idx in self.label_map.items()}
            self._loaded_model = True
        except Exception as exc:
            logger.warning(
                "Failed loading history sequence model %s: %s", model_path, exc
            )
            self.model = None
            self.label_map = {}
            self.index_to_label = {}
            self._loaded_model = False

    def _extract_tokens_from_h5(self, file_path: str) -> List[int]:
        return load_command_tokens_from_h5(
            file_path,
            vec_key=self.vec_key,
            command_col=self.command_col,
        )

    @staticmethod
    def _label_from_index(label_map: Dict[str, int], index: int) -> Optional[str]:
        for label, idx in label_map.items():
            if idx == int(index):
                return label
        return None

    def _sanitize_tokens_for_model(self, tokens: List[int]) -> List[int]:
        if not tokens:
            return [self.model_padding_idx]
        sanitized: List[int] = []
        vocab_upper = max(1, int(self.vocab_size))
        for token in tokens:
            token_id = int(token)
            if token_id < 0 or token_id >= vocab_upper:
                token_id = self.model_padding_idx
            sanitized.append(token_id)
        return sanitized

    def _predict_with_model(self, tokens: List[int]) -> Dict[str, Any]:
        if not (
            HAS_TORCH
            and self._loaded_model
            and self.model is not None
            and self.label_map
        ):
            return {"status": "model_unavailable"}

        assert torch is not None  # for type-checkers
        seq = self._sanitize_tokens_for_model(tokens[-self.max_model_sequence_length :])
        inputs = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([len(seq)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(inputs, lengths=lengths)
            probs = torch.softmax(logits, dim=1)[0]
            topk = min(2, int(probs.numel()))
            top_vals, top_idx = torch.topk(probs, k=topk)
            pred_idx = int(top_idx[0].item())
            label = self.index_to_label.get(pred_idx) or self._label_from_index(
                self.label_map, pred_idx
            )
            conf = float(top_vals[0].item())
            top2 = float(top_vals[1].item()) if topk > 1 else 0.0
            margin = conf - top2

        return {
            "status": "ok",
            "label": label,
            "confidence": conf,
            "top2_confidence": top2,
            "margin": margin,
            "source": "history_sequence_model",
            "label_map_size": len(self.label_map),
        }

    def _predict_with_prototypes(self, tokens: List[int]) -> Dict[str, Any]:
        if not self.prototype_scores:
            return {"status": "model_unavailable"}

        seq_len = len(tokens)
        if seq_len == 0:
            return {"status": "empty_sequence"}

        counts = Counter(tokens)
        bigram_counts = Counter(zip(tokens[:-1], tokens[1:])) if len(tokens) >= 2 else {}
        norm = float(seq_len)
        hist = {token: cnt / norm for token, cnt in counts.items()}
        bigram_norm = float(max(1, seq_len - 1))
        bigram_hist = {
            pair: cnt / bigram_norm for pair, cnt in dict(bigram_counts).items()
        }

        scores: List[Tuple[str, float]] = []
        for label, weights in self.prototype_scores.items():
            score = float(self.prototype_bias.get(label, 0.0))
            for token, weight in weights.items():
                score += (
                    hist.get(int(token), 0.0)
                    * float(weight)
                    * float(self.prototype_token_weight)
                )
            bigram_weights = self.prototype_bigram_scores.get(label, {})
            for pair, weight in bigram_weights.items():
                score += (
                    bigram_hist.get((int(pair[0]), int(pair[1])), 0.0)
                    * float(weight)
                    * float(self.prototype_bigram_weight)
                )
            scores.append((label, score))

        if not scores:
            return {"status": "model_unavailable"}

        scores.sort(key=lambda item: item[1], reverse=True)
        best_label, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else best_score - 1.0

        # Stable softmax for confidence.
        max_score = max(score for _, score in scores)
        exp_scores = [math.exp(score - max_score) for _, score in scores]
        denom = sum(exp_scores) or 1.0
        confidence = exp_scores[0] / denom
        top2_confidence = exp_scores[1] / denom if len(exp_scores) > 1 else 0.0
        margin = confidence - top2_confidence

        return {
            "status": "ok",
            "label": best_label,
            "confidence": float(confidence),
            "top2_confidence": float(top2_confidence),
            "margin": float(margin),
            "raw_score": float(best_score),
            "raw_score_gap": float(best_score - second_score),
            "source": "history_sequence_prototype",
            "label_map_size": len(scores),
            "prototype_token_weight": float(self.prototype_token_weight),
            "prototype_bigram_weight": float(self.prototype_bigram_weight),
        }

    def predict_from_tokens(self, tokens: List[int]) -> Dict[str, Any]:
        sequence: List[int] = []
        for token in tokens:
            try:
                token_id = int(token)
            except Exception:
                continue
            if token_id >= 0:
                sequence.append(token_id)
        seq_len = len(sequence)
        if seq_len == 0:
            return {
                "status": "empty_sequence",
                "label": None,
                "confidence": 0.0,
                "source": "history_sequence",
                "sequence_length": 0,
                "unique_commands": 0,
            }
        if seq_len < self.min_sequence_length:
            return {
                "status": "too_short",
                "label": None,
                "confidence": 0.0,
                "source": "history_sequence",
                "sequence_length": seq_len,
                "unique_commands": len(set(sequence)),
            }

        payload = self._predict_with_model(sequence)
        if payload.get("status") != "ok":
            payload = self._predict_with_prototypes(sequence)

        payload.setdefault("label", None)
        payload.setdefault("confidence", 0.0)
        payload.setdefault("source", "history_sequence")
        payload["sequence_length"] = seq_len
        payload["unique_commands"] = len(set(sequence))
        payload["sequence_summary"] = sequence_statistics(sequence, top_k=3)
        return payload

    def predict_from_h5_file(self, file_path: str) -> Dict[str, Any]:
        if not file_path:
            return {"status": "empty_input", "label": None, "confidence": 0.0}
        path = Path(file_path).expanduser()
        if not path.exists():
            return {"status": "file_not_found", "label": None, "confidence": 0.0}
        try:
            tokens = self._extract_tokens_from_h5(str(path))
        except Exception as exc:
            return {
                "status": "parse_error",
                "error": str(exc),
                "label": None,
                "confidence": 0.0,
            }

        payload = self.predict_from_tokens(tokens)
        payload["file_path"] = str(path)
        return payload


_HISTORY_SEQUENCE_CLASSIFIER: Optional[HistorySequenceClassifier] = None


def get_history_sequence_classifier() -> HistorySequenceClassifier:
    global _HISTORY_SEQUENCE_CLASSIFIER
    if _HISTORY_SEQUENCE_CLASSIFIER is None:
        _HISTORY_SEQUENCE_CLASSIFIER = HistorySequenceClassifier()
    return _HISTORY_SEQUENCE_CLASSIFIER


def reset_history_sequence_classifier() -> None:
    global _HISTORY_SEQUENCE_CLASSIFIER
    _HISTORY_SEQUENCE_CLASSIFIER = None
