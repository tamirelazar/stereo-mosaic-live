import os
import numpy as np
from smlive import utils as sol4_utils
from smlive import features
from smlive.volume import AlignedVolume
from smlive.errors import InsufficientTranslation


class Aligner:
    def __init__(self, frames_dir, file_prefix, num_images):
        self.frames_dir = frames_dir
        self.file_prefix = file_prefix
        self.files = [os.path.join(frames_dir, "%s%03d.jpg" % (file_prefix, i + 1))
                      for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))

    def align(self, translation_only=True, stabilize=False):
        pts_desc = []
        for f in self.files:
            image = sol4_utils.read_image(f, 1)
            h, w = image.shape
            pyr, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            pts_desc.append(features.find_features(pyr))

        Hs = []
        for i in range(len(pts_desc) - 1):
            p1, p2 = pts_desc[i][0], pts_desc[i + 1][0]
            d1, d2 = pts_desc[i][1], pts_desc[i + 1][1]
            ind1, ind2 = features.match_features(d1, d2, .7)
            p1, p2 = p1[ind1, :], p2[ind2, :]
            H12, _ = features.ransac_homography(p1, p2, 100, 6, translation_only)
            Hs.append(H12)

        accumulated = features.accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        homographies = np.stack(accumulated)
        used = features.filter_homographies_with_translation(homographies, minimum_right_translation=5)
        homographies = homographies[used]

        if stabilize:
            # Anchor every frame's y-translation to the reference frame (identity):
            # a lateral pan should have ~no vertical motion, so residual y-translation
            # is unwanted wobble that bows the mosaic's top edge. x-translation (the
            # parallax we want) is left untouched.
            homographies[:, 1, 2] = 0.0

        bounding_boxes = np.zeros((used.size, 2, 2))
        for i in range(used.size):
            bounding_boxes[i] = features.compute_bounding_box(homographies[i], w, h)
        global_offset = np.min(bounding_boxes, axis=(0, 1))
        bounding_boxes -= global_offset
        panorama_size = (np.max(bounding_boxes, axis=(0, 1)).astype(int) + 1)

        # per-frame accumulated x-translation in panorama coords (the camera path)
        path = bounding_boxes[:, 0, 0].copy()

        # Net translation must exceed one frame width for a valid mosaic.
        # (the original code crashed here with a bare `assert crop_left < crop_right`)
        crop_left = float(bounding_boxes[0][1, 0])    # right edge of first frame
        crop_right = float(bounding_boxes[-1][0, 0])  # left edge of last frame
        net_translation = float(path[-1] - path[0])
        if crop_left >= crop_right:
            raise InsufficientTranslation(
                "Net camera translation (%.0f px) is below one frame width (%d px): "
                "the first and last frames still overlap, so no mosaic can be formed. "
                "Try a longer lateral pan, a closer subject, or a higher-fps capture."
                % (net_translation, w))

        warnings = []
        # Near-pure-rotation / too-slow segments: large frame gaps with little x-gain.
        steps = np.diff(path)
        if steps.size and np.median(steps) < 1.0:
            warnings.append("Low per-frame translation detected; results may be soft. "
                            "A steadier, faster lateral pan improves sharpness.")

        return AlignedVolume(
            files=[self.files[i] for i in used],
            homographies=homographies,
            bounding_boxes=bounding_boxes,
            panorama_size=panorama_size,
            w=w, h=h,
            path=path,
            warnings=warnings,
        )
