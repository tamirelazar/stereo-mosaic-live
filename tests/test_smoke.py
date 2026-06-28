import numpy as np
from smlive import utils


def test_blur_spatial_preserves_shape_and_dtype():
    # build_gaussian_pyramid + blur_spatial are the carried primitives align depends on
    im = np.random.rand(64, 64).astype(np.float64)
    blurred = utils.blur_spatial(im, 3)
    assert blurred.shape == im.shape
    assert blurred.dtype == np.float64
