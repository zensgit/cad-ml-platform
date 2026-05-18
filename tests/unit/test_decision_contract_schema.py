"""Schema guard for the v1 classification decision contract.

Catches silent key drift in `build_classification_decision_contract` and its
helper `extract_label_decision_contract`: any rename, drop, or accidental key
addition flips the assertion before the change can reach assistant / batch /
benchmark consumers that read these keys positionally.
"""

from __future__ import annotations

from src.core.classification.decision_contract import (
    build_classification_decision_contract,
    extract_label_decision_contract,
)


EXTRACT_KEYS = frozenset(
    {
        "part_type",
        "fine_part_type",
        "coarse_part_type",
        "decision_source",
        "final_decision_source",
        "is_coarse_label",
    }
)


BUILD_KEYS = EXTRACT_KEYS | frozenset({"confidence_source", "rule_version"})


def test_extract_contract_returns_exactly_the_documented_key_set() -> None:
    contract = extract_label_decision_contract({})
    assert set(contract.keys()) == EXTRACT_KEYS, (
        f"extract_label_decision_contract key set drift: "
        f"got={sorted(contract.keys())} want={sorted(EXTRACT_KEYS)}"
    )


def test_build_contract_returns_exactly_the_documented_key_set() -> None:
    contract = build_classification_decision_contract({})
    assert set(contract.keys()) == BUILD_KEYS, (
        f"build_classification_decision_contract key set drift: "
        f"got={sorted(contract.keys())} want={sorted(BUILD_KEYS)}"
    )


def test_empty_payload_yields_all_none_or_default_values() -> None:
    contract = build_classification_decision_contract({})
    # Every documented key MUST be present, even when the payload is empty —
    # this is the contract that downstream consumers rely on.
    for key in BUILD_KEYS:
        assert key in contract, f"missing key {key!r}"
    # No accidental positive default values.
    assert contract["part_type"] is None
    assert contract["fine_part_type"] is None
    assert contract["coarse_part_type"] is None
    assert contract["decision_source"] is None
    assert contract["final_decision_source"] is None
    assert contract["confidence_source"] is None
    assert contract["rule_version"] is None
    assert contract["is_coarse_label"] is None


def test_fine_falls_back_to_part_type_when_unset() -> None:
    payload = {"part_type": "BRACKET"}
    contract = build_classification_decision_contract(payload)
    assert contract["part_type"] == "BRACKET"
    assert contract["fine_part_type"] == "BRACKET"


def test_coarse_part_type_derives_via_normalize_when_missing() -> None:
    # When the payload supplies only a fine label, the contract must still
    # populate `coarse_part_type` via normalize_coarse_label. The exact value
    # is left to the normalizer; we assert the *contract*: when fine is
    # present, coarse must be non-None or explicitly equal to fine.
    payload = {"fine_part_type": "BRACKET"}
    contract = extract_label_decision_contract(payload)
    assert contract["fine_part_type"] == "BRACKET"
    # coarse is either a normalized form, or falls back to fine itself —
    # either way, must NOT be None when fine is present and non-empty.
    assert contract["coarse_part_type"] is not None


def test_is_coarse_label_inferred_when_fine_equals_coarse() -> None:
    contract = extract_label_decision_contract(
        {"fine_part_type": "BRACKET", "coarse_part_type": "BRACKET"}
    )
    # Inferred True because fine == coarse and is_coarse_label was unset.
    assert contract["is_coarse_label"] is True


def test_is_coarse_label_explicit_false_is_preserved() -> None:
    contract = extract_label_decision_contract(
        {
            "fine_part_type": "BRACKET_L",
            "coarse_part_type": "BRACKET",
            "is_coarse_label": False,
        }
    )
    assert contract["is_coarse_label"] is False


def test_decision_source_first_present_wins() -> None:
    contract = extract_label_decision_contract(
        {
            "part_type": "X",
            "final_decision_source": "fusion",
            "decision_source": "hybrid",
            "confidence_source": "graph2d",
        }
    )
    assert contract["decision_source"] == "fusion"
    assert contract["final_decision_source"] == "fusion"


def test_decision_source_falls_back_to_legacy_key() -> None:
    contract = extract_label_decision_contract(
        {
            "part_type": "X",
            "confidence_source": "graph2d",
        }
    )
    assert contract["decision_source"] == "graph2d"


def test_build_contract_confidence_source_falls_back_to_decision_source() -> None:
    contract = build_classification_decision_contract(
        {"part_type": "X", "decision_source": "hybrid"}
    )
    assert contract["confidence_source"] == "hybrid"


def test_build_contract_rule_version_passthrough() -> None:
    contract = build_classification_decision_contract(
        {"part_type": "X", "rule_version": "v2.3"}
    )
    assert contract["rule_version"] == "v2.3"
