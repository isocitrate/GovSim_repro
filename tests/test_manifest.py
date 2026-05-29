from scripts.generate_manifest import PRESETS, rows


def test_full_manifest_has_360_trials():
    manifest = list(rows(num_trials=20, backend="vllm"))

    assert len(manifest) == 360
    assert {row["institution"] for row in manifest} == {
        "no_communication",
        "free_communication",
        "costly_punishment",
    }
    assert {row["game"] for row in manifest} == {"fishing", "sheep"}


def test_pilot_preset_has_54_trials():
    manifest = list(rows(num_trials=PRESETS["pilot"], backend="vllm"))

    assert len(manifest) == 54
