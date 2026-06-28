import os
import numpy as np
import pytest
from tests.data.make_fixture import make_sequence
from smlive.align import Aligner
from smlive.volume import AlignedVolume
from smlive.errors import InsufficientTranslation


def test_align_produces_volume_with_monotonic_path(tmp_path):
    d, prefix, n = make_sequence(str(tmp_path / "syn"), dx=30)
    vol = Aligner(d + os.sep, prefix, n).align(translation_only=True)
    assert isinstance(vol, AlignedVolume)
    assert vol.homographies.shape[0] == vol.path.shape[0]
    # lateral translation -> path is (weakly) monotonic in x
    assert np.all(np.diff(vol.path) >= -1e-6)


def test_insufficient_translation_raises_clear_error(tmp_path):
    # dx=0 -> camera never translates -> net translation below one frame width
    d, prefix, n = make_sequence(str(tmp_path / "static"), dx=0)
    with pytest.raises(InsufficientTranslation) as e:
        Aligner(d + os.sep, prefix, n).align(translation_only=True)
    assert "translation" in str(e.value).lower()
    # message must be actionable, not a traceback artifact
    assert any(w in str(e.value).lower() for w in ["longer", "closer", "fps"])
