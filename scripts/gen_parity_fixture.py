"""Emit tests-js/parity.json: a small manifest + expected centers/bounds per
(mode, viewpoint), computed via smlive's column_for_frame + apply_homography.
Single source of truth for JS<->Python parity."""
import os, json, numpy as np
from tests.data.make_fixture import make_sequence
from smlive.align import Aligner
from smlive.render import Renderer
from smlive import features

MODES = ["pushbroom", "xslit", "forward"]
VPS = [0.0, 0.5, 1.0]


def main():
    d, prefix, n = make_sequence("/tmp/_parity_syn", dx=30)
    vol = Aligner(d + os.sep, prefix, n).align(translation_only=True)
    r = Renderer(vol)
    W = int(vol.panorama_size[0])
    manifest = {
        "n": len(vol.files), "w": int(vol.w), "h": int(vol.h),
        "panorama_size": [W, int(vol.panorama_size[1])],
        "homographies": vol.homographies.tolist(),
        "bounding_boxes": vol.bounding_boxes.tolist(),
        "global_offset": [float(vol.global_offset[0]), float(vol.global_offset[1])],
    }
    expected = {}
    for mode in MODES:
        for vp in VPS:
            cen = []
            for i in range(len(vol.files)):
                col = r.column_for_frame(i, vp, mode)
                wx = features.apply_homography(np.array([[col, vol.h // 2]]),
                                               vol.homographies[i])[0, 0]
                cen.append(wx - vol.global_offset[0])
            cen = np.array(cen)
            bnd = (cen[:-1] + cen[1:]) / 2.0
            bnd = np.concatenate([[0.0], bnd, [float(W)]])
            bnd = np.clip(bnd, 0, W)
            expected["%s_%.1f" % (mode, vp)] = {"centers": cen.tolist(),
                                                "bounds": bnd.tolist()}
    os.makedirs("tests-js", exist_ok=True)
    json.dump({"manifest": manifest, "modes": MODES, "vps": VPS, "expected": expected},
              open("tests-js/parity.json", "w"))
    print("wrote tests-js/parity.json  n=%d" % len(vol.files))


if __name__ == "__main__":
    main()
