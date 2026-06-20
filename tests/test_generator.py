import pytest
from scenarios.generator import GeneratorConfig, ScenarioGenerator, _allocate


def test_generate_count():
    cfg = GeneratorConfig(n=20, seed=0)
    scenarios = ScenarioGenerator(cfg).generate()
    assert len(scenarios) == 20


def test_scene_ids_unique():
    cfg = GeneratorConfig(n=30, seed=1)
    scenarios = ScenarioGenerator(cfg).generate()
    ids = [s.scene_id for s in scenarios]
    assert len(ids) == len(set(ids))


def test_all_profiles_represented():
    cfg = GeneratorConfig(n=100, seed=42)
    scenarios = ScenarioGenerator(cfg).generate()
    profiles = {s.description for s in scenarios}
    expected = {"baseline", "crowded", "dark", "far", "occluded", "steep"}
    assert profiles == expected


def test_both_tiers_present_per_scene():
    cfg = GeneratorConfig(n=50, seed=42)
    scenarios = ScenarioGenerator(cfg).generate()
    multi = [s for s in scenarios if len(s.object_specs) >= 2]
    for s in multi:
        tiers = {spec.tier for spec in s.object_specs if spec.tier != "clutter"}
        assert "easy" in tiers and "hard" in tiers, f"{s.scene_id} missing a tier: {tiers}"


def test_easy_tier_class_names_are_coco():
    coco_names = {"cup", "bottle", "bowl", "teddy bear", "sports ball"}
    cfg = GeneratorConfig(n=50, seed=1)
    scenarios = ScenarioGenerator(cfg).generate()
    for s in scenarios:
        for spec in s.object_specs:
            if spec.tier == "easy":
                assert spec.class_name in coco_names, f"Unexpected easy name: {spec.class_name}"


def test_dark_profile_low_ambient():
    cfg = GeneratorConfig(n=100, seed=42)
    scenarios = ScenarioGenerator(cfg).generate()
    dark = [s for s in scenarios if s.description == "dark"]
    assert all(s.ambient_light <= 0.26 for s in dark)


def test_far_profile_large_distance():
    cfg = GeneratorConfig(n=100, seed=42)
    scenarios = ScenarioGenerator(cfg).generate()
    far = [s for s in scenarios if s.description == "far"]
    assert all(s.camera_distance >= 4.0 for s in far)


def test_occluded_profile_has_extra_objects():
    cfg = GeneratorConfig(n=100, seed=42)
    scenarios = ScenarioGenerator(cfg).generate()
    occ = [s for s in scenarios if s.description == "occluded"]
    # occluded scenes place primaries + occluders → at least 3 objects on average
    avg = sum(len(s.object_specs) for s in occ) / len(occ)
    assert avg >= 2.5


def test_allocate_sums_to_n():
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    counts = _allocate(100, weights)
    assert sum(counts.values()) == 100


def test_reproducible():
    cfg = GeneratorConfig(n=10, seed=99)
    s1 = ScenarioGenerator(cfg).generate()
    s2 = ScenarioGenerator(cfg).generate()
    assert [s.scene_id for s in s1] == [s.scene_id for s in s2]
    assert [s.camera_distance for s in s1] == [s.camera_distance for s in s2]
