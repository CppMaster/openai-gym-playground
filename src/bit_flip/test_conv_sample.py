import pytest
import numpy as np

from conv_sample import get_accuracy


@pytest.mark.parametrize(("pred", "target", "output"), [
    (np.array([[[True]]]), np.array([[[True]]]), 1.0),
    (np.array([[[True]]]), np.array([[[False]]]), 0.0),
    (np.array([[[True]], [[True]]]), np.array([[[True]], [[False]]]), 0.5),
    (np.array([[[True], [False]]]), np.array([[[False], [False]]]), 0.0)
])
def test_get_accuracy(pred: np.ndarray, target: np.ndarray, output: float):
    assert get_accuracy(pred, target) == output
