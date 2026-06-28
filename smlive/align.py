import os
import numpy as np
from smlive import utils as sol4_utils
from smlive import features
from smlive.volume import AlignedVolume


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
            self.h, self.w = image.shape
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

        bounding_boxes = np.zeros((used.size, 2, 2))
        for i in range(used.size):
            bounding_boxes[i] = features.compute_bounding_box(homographies[i], self.w, self.h)
        global_offset = np.min(bounding_boxes, axis=(0, 1))
        bounding_boxes -= global_offset
        panorama_size = (np.max(bounding_boxes, axis=(0, 1)).astype(int) + 1)

        # per-frame accumulated x-translation in panorama coords (the camera path)
        path = bounding_boxes[:, 0, 0].copy()

        return AlignedVolume(
            files=[self.files[i] for i in used],
            homographies=homographies,
            bounding_boxes=bounding_boxes,
            panorama_size=panorama_size,
            w=self.w, h=self.h,
            path=path,
            warnings=[],
        )
