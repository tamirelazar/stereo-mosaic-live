"""Export the bundled boat demo asset into docs/asset/ for the web viewer.
Reads the locally-extracted boat frames in examples/_frames (gitignored). Run
once; the resulting docs/asset/ IS committed because the source frames/video
are not. (To recreate examples/_frames from videos/boat.mp4: smlive's ffmpeg
extraction, or the Phase 1 gen path.)"""
import os
from smlive.align import Aligner
from smlive.render import Renderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "examples", "_frames") + os.sep  # 450 extracted boat frames
OUT = os.path.join(ROOT, "docs", "asset")
STRIDE = 2  # 450 -> ~225 frames; ~4.5 MB of JPGs


def main():
    vol = Aligner(FRAMES, "boat", 450).align(translation_only=True, stabilize=False)
    Renderer(vol).export_web_asset(OUT, max_height=240, stride=STRIDE)
    print("exported", len([f for f in os.listdir(OUT) if f.endswith(".jpg")]), "frames to", OUT)


if __name__ == "__main__":
    main()
