from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ml.history_sequence_classifier import HistorySequenceClassifier


def test_history_sequence_classifier_uses_prototypes_and_returns_summary(
    tmp_path: Path,
) -> None:
    prototypes = {
        "labels": {
            "bracket": {
                "token_weights": {"1": 3.0, "2": 1.0},
                "bigram_weights": {"1,2": 2.0},
                "bias": 0.2,
            },
            "plate": {
                "token_weights": {"8": 2.0},
                "bigram_weights": {"8,8": 1.0},
            },
        }
    }
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")

    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
        min_sequence_length=2,
    )

    result = classifier.predict_from_tokens([1, 2, 1, 3])

    assert result["status"] == "ok"
    assert result["label"] == "bracket"
    assert result["source"] == "history_sequence_prototype"
    assert result["sequence_length"] == 4
    assert result["unique_commands"] == 3
    assert result["sequence_summary"]["top_commands"][0] == (1, 2)


def test_history_sequence_classifier_loads_checkpoint_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from src.ml.train.sequence_encoder import SequenceCommandClassifier

    checkpoint_path = tmp_path / "history.ckpt"
    model = SequenceCommandClassifier(
        vocab_size=16,
        num_classes=2,
        embedding_dim=8,
        hidden_dim=8,
        dropout=0.0,
        padding_idx=0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 2.5]))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": {"alpha": 0, "beta": 1},
            "model_config": {
                "vocab_size": 16,
                "embedding_dim": 8,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "padding_idx": 0,
                "max_sequence_length": 16,
            },
        },
        checkpoint_path,
    )

    classifier = HistorySequenceClassifier(
        prototypes_path=str(tmp_path / "missing.json"),
        model_path=str(checkpoint_path),
        min_sequence_length=2,
    )

    result = classifier.predict_from_tokens([1, 2, 3])

    assert result["status"] == "ok"
    assert result["label"] == "beta"
    assert result["source"] == "history_sequence_model"
    assert result["label_map_size"] == 2


def test_history_sequence_classifier_reads_h5_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.h5"
    file_path.write_bytes(b"placeholder")

    prototypes = {
        "labels": {
            "bracket": {"token_weights": {"1": 2.0}},
        }
    }
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")
    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        min_sequence_length=2,
    )
    classifier._extract_tokens_from_h5 = lambda _path: [1, 2, 1]

    result = classifier.predict_from_h5_file(str(file_path))

    assert result["status"] == "ok"
    assert result["file_path"] == str(file_path)
    assert result["sequence_length"] == 3


def test_history_sequence_classifier_returns_too_short(tmp_path: Path) -> None:
    prototypes = {
        "labels": {
            "shaft": {"token_weights": {"6": 0.4, "10": 0.6}},
        }
    }
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")
    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
        min_sequence_length=5,
    )

    result = classifier.predict_from_tokens([6, 10])

    assert result["status"] == "too_short"
    assert result["label"] is None


def test_history_sequence_classifier_handles_missing_h5_file(tmp_path: Path) -> None:
    prototypes = {"labels": {"shaft": {"token_weights": {"6": 1.0}}}}
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")
    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
    )

    result = classifier.predict_from_h5_file(str(tmp_path / "missing.h5"))

    assert result["status"] == "file_not_found"
    assert result["label"] is None


def test_history_sequence_classifier_ignores_non_numeric_tokens(
    tmp_path: Path,
) -> None:
    prototypes = {"labels": {"shaft": {"token_weights": {"6": 0.4, "10": 0.6}}}}
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")
    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
        min_sequence_length=2,
    )

    result = classifier.predict_from_tokens([6, "x", None, 10, -1])  # type: ignore[list-item]

    assert result["status"] == "ok"
    assert result["label"] == "shaft"
    assert result["sequence_length"] == 2


def test_history_sequence_classifier_sanitizes_model_tokens(tmp_path: Path) -> None:
    prototypes = {"labels": {"shaft": {"token_weights": {"6": 1.0}}}}
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")
    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
    )

    class StubModel:
        def __call__(self, inputs, lengths=None):  # noqa: ANN001, ANN201
            assert int(inputs.max().item()) <= 3
            return torch.tensor([[0.1, 1.5]], dtype=torch.float32)

    torch = pytest.importorskip("torch")
    classifier.model = StubModel()
    classifier._loaded_model = True
    classifier.vocab_size = 4
    classifier.model_padding_idx = 0
    classifier.label_map = {"shaft": 0, "link": 1}
    classifier.index_to_label = {0: "shaft", 1: "link"}

    result = classifier.predict_from_tokens([999, 1, 2, 3])

    assert result["status"] == "ok"
    assert result["label"] == "link"
    assert result["source"] == "history_sequence_model"


def test_history_sequence_classifier_uses_bigram_weights(tmp_path: Path) -> None:
    prototypes = {
        "labels": {
            "A": {
                "token_weights": {"1": 0.1, "2": 0.1},
                "bigram_weights": {"1,2": 1.2, "2,1": 0.1},
            },
            "B": {
                "token_weights": {"1": 0.1, "2": 0.1},
                "bigram_weights": {"2,1": 1.2, "1,2": 0.1},
            },
        }
    }
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")
    classifier = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
        min_sequence_length=2,
    )

    result = classifier.predict_from_tokens([1, 2, 1, 2, 1, 2])

    assert result["status"] == "ok"
    assert result["label"] == "A"


def test_history_sequence_classifier_supports_bigram_weight_tuning(
    tmp_path: Path,
) -> None:
    prototypes = {
        "labels": {
            "A": {
                "token_weights": {"1": 0.0, "2": 0.0},
                "bigram_weights": {"1,2": 2.0},
            },
            "B": {
                "token_weights": {"1": 0.3, "2": 0.3},
                "bigram_weights": {},
            },
        }
    }
    prototype_path = tmp_path / "prototypes.json"
    prototype_path.write_text(json.dumps(prototypes), encoding="utf-8")

    no_bigram = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
        min_sequence_length=2,
        prototype_bigram_weight=0.0,
    )
    with_bigram = HistorySequenceClassifier(
        prototypes_path=str(prototype_path),
        model_path="",
        min_sequence_length=2,
        prototype_bigram_weight=1.0,
    )

    result_without = no_bigram.predict_from_tokens([1, 2, 1, 2, 1, 2])
    result_with = with_bigram.predict_from_tokens([1, 2, 1, 2, 1, 2])

    assert result_without["status"] == "ok"
    assert result_without["label"] == "B"
    assert result_with["status"] == "ok"
    assert result_with["label"] == "A"
