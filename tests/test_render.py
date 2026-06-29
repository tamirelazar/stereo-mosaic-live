import os
import json
import numpy as np
from tests.data.make_fixture import make_sequence
from smlive.align import Aligner
from smlive.render import Renderer


def _vol(tmp_path):
    d, prefix, n = make_sequence(str(tmp_path / "syn"), dx=30)
    return Aligner(d + os.sep, prefix, n).align(translation_only=True)


def test_pushbroom_is_constant_column_across_frames(tmp_path):
    r = Renderer(_vol(tmp_path))
    cols = [r.column_for_frame(i, viewpoint=0.5, mode="pushbroom") for i in range(5)]
    assert np.allclose(cols, cols[0])  # pushbroom samples the SAME column every frame


def test_xslit_column_varies_linearly_with_frame(tmp_path):
    r = Renderer(_vol(tmp_path))
    cols = np.array([r.column_for_frame(i, viewpoint=0.5, mode="xslit") for i in range(5)])
    diffs = np.diff(cols)
    assert not np.allclose(diffs, 0)              # it MOVES (unlike pushbroom)
    assert np.allclose(diffs, diffs[0])           # and moves linearly


def test_forward_is_not_pushbroom_at_default_viewpoint(tmp_path):
    # regression: forward must NOT collapse to pushbroom at the default viewpoint=0.5
    r = Renderer(_vol(tmp_path))
    fwd = np.array([r.column_for_frame(i, viewpoint=0.5, mode="forward") for i in range(5)])
    pb = np.array([r.column_for_frame(i, viewpoint=0.5, mode="pushbroom") for i in range(5)])
    assert not np.allclose(fwd, fwd[0])   # forward genuinely varies across frames
    assert not np.allclose(fwd, pb)       # and is distinct from pushbroom
    # forward is quadratic (not linear) -> non-constant first-differences, unlike xslit
    diffs = np.diff(fwd)
    assert not np.allclose(diffs, diffs[0])


def test_forward_mode_renders_nonblank(tmp_path):
    # closes the forward end-to-end render coverage gap
    r = Renderer(_vol(tmp_path))
    pano = r.render(viewpoint=0.5, mode="forward")
    assert pano.ndim == 3 and pano.shape[2] == 3
    assert pano.min() >= 0.0 and pano.max() <= 1.0
    assert pano.max() > 0.01                          # real content, not a black canvas
    assert (pano.sum(axis=(0, 2)) > 0).mean() > 0.5   # forward sweep fills a substantial width


def test_render_returns_normalized_rgb(tmp_path):
    r = Renderer(_vol(tmp_path))
    pano = r.render(viewpoint=0.5, mode="xslit")
    assert pano.ndim == 3 and pano.shape[2] == 3
    assert pano.min() >= 0.0 and pano.max() <= 1.0
    assert pano.max() > 0.01                       # strips actually composited, not a black canvas
    col_filled = (pano.sum(axis=(0, 2)) > 0).mean()  # fraction of canvas columns with content
    assert col_filled > 0.8                          # mosaic fills (almost) the full width


def test_blending_removes_hard_seam_discontinuity(tmp_path):
    r = Renderer(_vol(tmp_path))
    hard = r.render(viewpoint=0.5, mode="pushbroom", blend=False)
    soft = r.render(viewpoint=0.5, mode="pushbroom", blend=True)
    # total horizontal gradient energy should drop with feathering
    def hgrad(p): return np.abs(np.diff(p, axis=1)).sum()
    assert hgrad(soft) <= hgrad(hard)


def test_export_web_asset_writes_frames_and_manifest(tmp_path):
    r = Renderer(_vol(tmp_path))
    out = tmp_path / "asset"
    r.export_web_asset(str(out), max_height=60)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n"] == len(r.v.files)
    assert manifest["scale"] <= 1.0
    assert (out / "frame0001.jpg").exists()
    assert "panorama_size" in manifest and len(manifest["panorama_size"]) == 2
    for key in ["w", "h", "scale", "n", "panorama_size", "centers_pushbroom",
                "homographies", "bounding_boxes", "warnings", "global_offset"]:
        assert key in manifest, "manifest missing key: %s" % key
    assert len(manifest["centers_pushbroom"]) == manifest["n"]
    assert len(manifest["global_offset"]) == 2
    assert all(isinstance(x, float) for x in manifest["global_offset"])


def test_export_web_asset_stride_subsamples(tmp_path):
    r = Renderer(_vol(tmp_path))
    out = tmp_path / "asset_stride"
    r.export_web_asset(str(out), max_height=60, stride=2)
    manifest = json.loads((out / "manifest.json").read_text())
    expected_n = (len(r.v.files) + 1) // 2
    assert manifest["n"] == expected_n
    assert len(manifest["centers_pushbroom"]) == expected_n
    # written frame count agrees with n
    written = sorted(p.name for p in out.glob("frame*.jpg"))
    assert len(written) == expected_n
    assert written[0] == "frame0001.jpg"
