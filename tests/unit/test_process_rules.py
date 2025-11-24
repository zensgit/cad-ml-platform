from src.core.process_rules import recommend


def test_process_rules_default_steel_low_volume():
    r = recommend(material="steel", complexity="low", volume=1000.0)
    assert r["primary"] in {"cnc_machining", "die_casting"}


def test_process_rules_fallback():
    r = recommend(material="unknown_material", complexity="low", volume=1e9)
    assert r["primary"] == "cnc_machining" or r["primary"] == "casting"

