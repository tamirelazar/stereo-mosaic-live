import numpy as np
from smlive import utils as sol4_utils
from smlive import features


class Renderer:
    def __init__(self, volume):
        self.v = volume

    def column_for_frame(self, i, viewpoint, mode):
        """Source column (in frame i's own pixel coords) to center the strip on.

        pushbroom: same column every frame (a fixed slit) -> the showcase behavior.
        xslit:     column slides linearly with frame index -> the virtual second slit,
                   `viewpoint` shifts which depth plane is fronto-parallel.
        forward:   column slides nonlinearly (quadratic) -> simulates forward motion.
        """
        n = len(self.v.files)
        base = viewpoint * self.v.w
        if mode == "pushbroom":
            return base
        t = i / max(n - 1, 1)
        span = self.v.w * 0.5  # how far the slit sweeps across the frame
        if mode == "xslit":
            return base + (t - 0.5) * span
        if mode == "forward":
            return base + (t - 0.5) ** 2 * np.sign(viewpoint - 0.5) * span
        raise ValueError("unknown mode: %s" % mode)

    def render(self, viewpoint=0.5, mode="xslit", blend=True, feather=8):
        v = self.v
        pano_w, pano_h = int(v.panorama_size[0]), int(v.panorama_size[1])

        # warped x-position of each frame's chosen source column, in panorama coords
        centers = np.zeros(len(v.files))
        for i in range(len(v.files)):
            col = self.column_for_frame(i, viewpoint, mode)
            warped = features.apply_homography(
                np.array([[col, v.h // 2]]), v.homographies[i])
            centers[i] = warped[0, 0] - v.global_offset[0]

        # strip boundaries = midpoints between consecutive centers
        bounds = (centers[:-1] + centers[1:]) / 2.0
        bounds = np.concatenate([[0.0], bounds, [pano_w]]).round().astype(int)
        bounds = np.clip(bounds, 0, pano_w)

        acc = np.zeros((pano_h, pano_w, 3), dtype=np.float64)
        wsum = np.zeros((pano_h, pano_w, 1), dtype=np.float64)
        for i in range(len(v.files)):
            image = sol4_utils.read_image(v.files[i], 2)
            warped_image = features.warp_image(image, v.homographies[i])
            x_off, y_off = v.bounding_boxes[i][0].astype(int)
            y_bottom = y_off + warped_image.shape[0]
            x0 = max(bounds[i] - (feather if blend else 0), 0)
            x1 = min(bounds[i + 1] + (feather if blend else 0), pano_w)
            if x1 <= x0:
                continue
            strip = warped_image[:, x0 - x_off:x1 - x_off]
            sw = strip.shape[1]
            ramp = np.ones(sw)
            if blend and feather > 0:
                f = min(feather, sw // 2)
                if f > 0:
                    if i > 0:  # blend left edge only when there is a left neighbour
                        ramp[:f] = np.linspace(0, 1, f)
                    if i < len(v.files) - 1:  # blend right edge only when there is a right neighbour
                        ramp[-f:] = np.linspace(1, 0, f)
            wmask = ramp[None, :, None]
            acc[y_off:y_bottom, x0:x0 + sw] += strip * wmask
            wsum[y_off:y_bottom, x0:x0 + sw] += wmask
        pano = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)
        return pano.clip(0, 1)
