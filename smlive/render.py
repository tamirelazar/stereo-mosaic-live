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

    def render(self, viewpoint=0.5, mode="xslit"):
        v = self.v
        pano_w, pano_h = int(v.panorama_size[0]), int(v.panorama_size[1])
        pano = np.zeros((pano_h, pano_w, 3), dtype=np.float64)

        # warped x-position of each frame's chosen source column, in panorama coords
        centers = np.zeros(len(v.files))
        for i in range(len(v.files)):
            col = self.column_for_frame(i, viewpoint, mode)
            warped = features.apply_homography(
                np.array([[col, v.h // 2]]), v.homographies[i])
            centers[i] = warped[0, 0] - 0  # global_offset already folded into bounding_boxes
        centers -= float(np.min(v.bounding_boxes[:, 0, 0]))  # align to canvas origin

        # strip boundaries = midpoints between consecutive centers
        bounds = (centers[:-1] + centers[1:]) / 2.0
        bounds = np.concatenate([[0.0], bounds, [pano_w]]).round().astype(int)
        bounds = np.clip(bounds, 0, pano_w)

        for i in range(len(v.files)):
            image = sol4_utils.read_image(v.files[i], 2)
            warped_image = features.warp_image(image, v.homographies[i])
            x_off, y_off = v.bounding_boxes[i][0].astype(int)
            y_bottom = y_off + warped_image.shape[0]
            x0, x1 = bounds[i], bounds[i + 1]
            if x1 <= x0:
                continue
            strip = warped_image[:, x0 - x_off:x1 - x_off]
            pano[y_off:y_bottom, x0:x0 + strip.shape[1]] = strip
        return pano.clip(0, 1)
