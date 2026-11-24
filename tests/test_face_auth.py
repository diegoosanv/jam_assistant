import numpy as np
import pytest

from jam.vision.face_auth import CV2_AVAILABLE, JamFaceAuth


def _make_synthetic_face(seed: int, brightness_shift: int = 0) -> np.ndarray:
    """Create a deterministic structured 128x128 grayscale face-like pattern for tests."""
    rng = np.random.default_rng(seed)
    img = np.full((128, 128), 110, dtype=np.int16)

    # Distinct structured regions (forehead, eyes, nose bridge, mouth, jawline)
    eye_y = 40 + int(rng.integers(-5, 6))
    eye_gap = 22 + int(rng.integers(-4, 5))
    img[eye_y - 6 : eye_y + 6, 64 - eye_gap - 8 : 64 - eye_gap + 8] = 35
    img[eye_y - 6 : eye_y + 6, 64 + eye_gap - 8 : 64 + eye_gap + 8] = 35

    nose_w = 5 + int(rng.integers(0, 5))
    img[50:84, 64 - nose_w : 64 + nose_w] = 175

    mouth_y = 95 + int(rng.integers(-4, 5))
    mouth_w = 18 + int(rng.integers(-4, 6))
    img[mouth_y - 4 : mouth_y + 4, 64 - mouth_w : 64 + mouth_w] = 45

    # Unique per-identity texture pattern
    texture = rng.integers(-22, 23, size=(16, 16), dtype=np.int16)
    texture_up = np.repeat(np.repeat(texture, 8, axis=0), 8, axis=1)
    img = img + texture_up + brightness_shift
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
def test_face_auth_enroll_recognize_and_reject(tmp_path):
    auth = JamFaceAuth(faces_dir=str(tmp_path / "faces"))
    assert auth.is_available() is True
    assert auth.get_status()["profiles_count"] == 0

    # Enroll two distinct synthetic identities
    diego_samples = [
        _make_synthetic_face(seed=101, brightness_shift=-10),
        _make_synthetic_face(seed=101, brightness_shift=0),
        _make_synthetic_face(seed=101, brightness_shift=12),
    ]
    ana_samples = [
        _make_synthetic_face(seed=202, brightness_shift=-8),
        _make_synthetic_face(seed=202, brightness_shift=5),
        _make_synthetic_face(seed=202, brightness_shift=15),
    ]

    r1 = auth.enroll_from_faces("Diego", diego_samples)
    r2 = auth.enroll_from_faces("Ana", ana_samples)
    assert r1["success"] is True and r1["samples"] == 3
    assert r2["success"] is True and r2["samples"] == 3
    assert auth.get_status()["profiles_count"] == 2

    # Reload from disk to verify persistence
    reloaded = JamFaceAuth(faces_dir=str(tmp_path / "faces"))
    assert set(reloaded.get_status()["enrolled_users"]) == {"Diego", "Ana"}

    # Recognize Diego under slightly shifted lighting
    probe_diego = _make_synthetic_face(seed=101, brightness_shift=20)
    match = reloaded.recognize_from_face(probe_diego, threshold=0.72)
    assert match["recognized"] is True
    assert match["user"] == "Diego"
    assert match["confidence"] >= 0.72

    # Recognize Ana
    probe_ana = _make_synthetic_face(seed=202, brightness_shift=-15)
    match_ana = reloaded.recognize_from_face(probe_ana, threshold=0.72)
    assert match_ana["recognized"] is True
    assert match_ana["user"] == "Ana"

    # Unknown person should NOT be falsely accepted with high threshold
    stranger = _make_synthetic_face(seed=999, brightness_shift=0)
    stranger_res = reloaded.recognize_from_face(stranger, threshold=0.85)
    assert stranger_res["recognized"] is False

    # Delete single profile and reset
    assert reloaded.delete_profile("Ana") == 1
    assert reloaded.get_status()["enrolled_users"] == ["Diego"]
    assert reloaded.delete_profile(None) == 1
    assert reloaded.get_status()["profiles_count"] == 0


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
def test_face_auth_handles_missing_file_gracefully(tmp_path):
    auth = JamFaceAuth(faces_dir=str(tmp_path / "faces"))
    res = auth.enroll_from_image_file("Diego", str(tmp_path / "does_not_exist.jpg"))
    assert res["success"] is False
    rec = auth.recognize_from_image_file(str(tmp_path / "does_not_exist.jpg"))
    assert rec["recognized"] is False
    assert rec["reason"] == "file_not_found"
