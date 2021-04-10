import matplotlib.pyplot as plt
import numpy as np
from cirq.sim.density_matrix_utils import _probs, _validate_num_qubits, _validate_density_matrix_qid_shape
from typing import List, Optional, Tuple


class Partial_Trace:
	def __init__(self, state: np.array, qubits_out: int):

		self.state = state
		self.m = qubits_out

		if self.state.ndim == 1:
			self.state = np.outer(self.state, self.state)

		self.n, _ = self.state.shape

		self.basis_b = [_ for _ in np.identity(int(2**self.m))]
		self.basis_a = [_ for _ in np.identity(int(self.n / 2**self.m))]

	def get_entry(self, i, j, basis_a, basis_b, state):
		sigma = 0
		for k in range(self.m):
			ab = np.kron(basis_a[i], basis_b[k])
			ba = np.kron(basis_a[j], basis_b[k])
			right_side = np.dot(state, ba)
			sigma += np.inner(ab, right_side)
		return sigma

	def compute_matrix(self):
		a = [i for i in range(2**self.m)]
		b = [i for i in range(2**self.m)]
		entries_pre = [(x, y) for x in a for y in b]

		entries = []
		for i, j in entries_pre:
			entries.append(self.get_entry(i, j, self.basis_a, self.basis_b, self.state))
		entries = np.array(entries)

		return entries.reshape(2**self.m, 2**self.m)


def get_pixel_value_ket0(
		density_matrix: np.ndarray,
		indices: List[int],
		*,  # Force keyword arguments
		qid_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
	if qid_shape is None:
		num_qubits = _validate_num_qubits(density_matrix)
		qid_shape = (2,) * num_qubits
	else:
		_validate_density_matrix_qid_shape(density_matrix, qid_shape)

	return _probs(density_matrix, indices, qid_shape)[0]


def compute_image_2x1(pixels: List):
	pixels = np.array(pixels)
	shape = pixels.shape
	if len(shape) % 2 != 0:
		raise ValueError('Matrix was not square. Shape was {}'.format(shape))
	return plt.imshow(pixels, cmap='gray')
