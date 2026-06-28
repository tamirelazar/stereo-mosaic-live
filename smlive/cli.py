import argparse
import os
import subprocess
import sys
from imageio import imwrite
from smlive.align import Aligner
from smlive.render import Renderer
from smlive.errors import InsufficientTranslation


def build_parser():
    p = argparse.ArgumentParser(prog="smlive", description="Choose your viewpoint after the fact.")
    p.add_argument("--input", required=True, help="input video file")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--mode", default="xslit", choices=["pushbroom", "xslit", "forward"])
    p.add_argument("--viewpoint", type=float, default=0.5, help="0..1 fraction of frame width")
    p.add_argument("--frames", type=int, default=0, help="max frames (0 = all extracted)")
    p.add_argument("--stabilize", action="store_true")
    p.add_argument("--web-asset", action="store_true", help="also export the web viewer asset")
    return p


def run_from_frames(frames_dir, prefix, n, mode, viewpoint, out_dir, stabilize, web_asset=False):
    os.makedirs(out_dir, exist_ok=True)
    try:
        vol = Aligner(frames_dir, prefix, n).align(translation_only=True, stabilize=stabilize)
    except InsufficientTranslation as e:
        print("error: %s" % e)
        return 2
    for w in vol.warnings:
        print("warning: %s" % w)
    r = Renderer(vol)
    pano = r.render(viewpoint=viewpoint, mode=mode)
    imwrite(os.path.join(out_dir, "mosaic_%s.png" % mode),
            (pano.clip(0, 1) * 255).astype("uint8"))
    if web_asset:
        r.export_web_asset(os.path.join(out_dir, "web"))
    return 0


def _extract_frames(video, frames_dir, prefix):
    os.makedirs(frames_dir, exist_ok=True)
    # Use subprocess.run with a list to avoid shell injection from user-supplied paths.
    out_pattern = os.path.join(frames_dir, "%s%%03d.jpg" % prefix)
    subprocess.run(["ffmpeg", "-i", video, out_pattern], check=False)
    return len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])


def main(argv=None):
    args = build_parser().parse_args(argv)
    prefix = os.path.splitext(os.path.basename(args.input))[0]
    frames_dir = os.path.join(args.out, "frames")
    n = _extract_frames(args.input, frames_dir, prefix)
    if n == 0:
        print("error: no frames extracted — is ffmpeg installed and the input path valid?")
        return 2
    if args.frames:
        n = min(n, args.frames)
    return run_from_frames(frames_dir + os.sep, prefix, n, args.mode,
                           args.viewpoint, args.out, args.stabilize, args.web_asset)


if __name__ == "__main__":
    sys.exit(main())
