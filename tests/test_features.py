import numpy as np
from smlive import features


def test_apply_homography_translation():
    H = np.eye(3)
    H[0, 2] = 10.0  # +10 in x
    H[1, 2] = -5.0  # -5 in y
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = features.apply_homography(pts, H)
    assert np.allclose(out, np.array([[10.0, -5.0], [13.0, -1.0]]))


def test_estimate_rigid_transform_translation_only_recovers_shift():
    pts1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    pts2 = pts1 + np.array([7.0, -3.0])
    H = features.estimate_rigid_transform(pts1, pts2, translation_only=True)
    assert np.allclose(H[:2, 2], np.array([7.0, -3.0]))
    assert np.allclose(H[:2, :2], np.eye(2))
