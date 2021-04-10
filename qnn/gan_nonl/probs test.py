import numpy as np
from cirq.sim.density_matrix_utils import _probs, _validate_num_qubits, _validate_density_matrix_qid_shape
from typing import List, Optional, Tuple


def probs_density_matrix(
        density_matrix: np.ndarray,
        indices: List[int],
        *,  # Force keyword arguments
        qid_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    if qid_shape is None:
        num_qubits = _validate_num_qubits(density_matrix)
        qid_shape = (2,) * num_qubits
    else:
        _validate_density_matrix_qid_shape(density_matrix, qid_shape)

    return _probs(density_matrix, indices, qid_shape)


probs = probs_density_matrix(np.outer([0.707 + 0j, 0.707 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                                      [0.707 + 0j, 0.707 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]), [2])
print(probs[0])
