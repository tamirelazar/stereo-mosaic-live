import os
import numpy as np
from imageio import imwrite
from smlive import cli


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
