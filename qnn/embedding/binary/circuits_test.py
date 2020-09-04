from qnn.embedding.binary.circuits_binary import Circuits, print_circuits, print_vectors, sample
from qnn.embedding.generate_data import GenerateRNBinary
import numpy as np
import cirq

gd_func = GenerateRNBinary(total_data_points=20, deviation=.2)
dict, data_num = gd_func.gen()
cs = Circuits(dict.values[0], depth=4).binary()
theta= np.random.normal(0, 2*np.pi, 120)

vectors = sample(cs, theta)
print(vectors)
