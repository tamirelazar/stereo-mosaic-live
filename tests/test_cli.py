import os
import shutil
from pathlib import Path
import numpy as np
import pytest
from imageio.v2 import imread
from smlive import cli


HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_parser_defaults():
    args = cli.build_parser().parse_args(["--input", "v.mp4", "--out", "o"])
    assert args.mode == "xslit"
    assert args.viewpoint == 0.5


def test_run_from_frames_dir_writes_mosaic(tmp_path):
    from tests.data.make_fixture import make_sequence
    d, prefix, n = make_sequence(str(tmp_path / "syn"), dx=30)
    out = tmp_path / "out"
    rc = cli.run_from_frames(d + os.sep, prefix, n, mode="pushbroom",
                             viewpoint=0.5, out_dir=str(out), stabilize=False)
    assert rc == 0
    assert (out / "mosaic_pushbroom.png").exists()


def test_insufficient_translation_exits_clean(tmp_path, capsys):
    from tests.data.make_fixture import make_sequence
    d, prefix, n = make_sequence(str(tmp_path / "static"), dx=0)
    out = tmp_path / "out"
    rc = cli.run_from_frames(d + os.sep, prefix, n, mode="xslit",
                             viewpoint=0.5, out_dir=str(out), stabilize=False)
    assert rc != 0
    assert "translation" in capsys.readouterr().out.lower()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_main_guards_zero_frames(tmp_path, capsys):
    """main() prints a clear error and returns 2 when ffmpeg extracts 0 frames."""
    rc = cli.main(["--input", "nonexistent_file.mp4", "--out", str(tmp_path)])
    assert rc == 2
    out = capsys.readouterr().out.lower()
    assert "error" in out


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_e2e_boat_produces_nonempty_mosaic(tmp_path):
    rc = cli.main(["--input", "videos/boat.mp4", "--out", str(tmp_path),
                   "--mode", "xslit", "--frames", "150"])
    assert rc == 0
    pano = imread(tmp_path / "mosaic_xslit.png")
    assert pano.shape[0] > 0 and pano.shape[1] > pano.shape[0]  # wide mosaic
    assert pano.std() > 5  # not a blank frame


def test_golden_regression_synthetic(tmp_path):
    from tests.data.make_fixture import make_sequence
    d, prefix, n = make_sequence(str(tmp_path / "syn"), seed=0, dx=30)
    from smlive.align import Aligner
    from smlive.render import Renderer
    vol = Aligner(d + os.sep, prefix, n).align(translation_only=True)
    pano = Renderer(vol).render(viewpoint=0.5, mode="pushbroom", blend=False)
    golden_path = str(Path(__file__).parent / "data" / "golden" / "syn_pushbroom.npy")
    if not os.path.exists(golden_path):
        os.makedirs(os.path.dirname(golden_path), exist_ok=True)
        np.save(golden_path, pano)
        pytest.skip("golden created; re-run to compare")
    golden = np.load(golden_path)
    assert pano.shape == golden.shape
    assert np.abs(pano - golden).mean() < 1e-6
