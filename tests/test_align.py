import os
import numpy as np
from tests.data.make_fixture import make_sequence
from smlive.align import Aligner
from smlive.volume import AlignedVolume


def test_align_produces_volume_with_monotonic_path(tmp_path):
    d, prefix, n = make_sequence(str(tmp_path / "syn"))
    vol = Aligner(d + os.sep, prefix, n).align(translation_only=True)
    assert isinstance(vol, AlignedVolume)
    assert vol.homographies.shape[0] == vol.path.shape[0]
    # lateral translation -> path is (weakly) monotonic in x
    assert np.all(np.diff(vol.path) >= -1e-6)
