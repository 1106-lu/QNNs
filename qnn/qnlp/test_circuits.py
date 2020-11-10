import numpy as np

from qnn.qnlp.circuits_words import CircuitsWords

c = CircuitsWords('C:/Users/usuario/Desktop/QIT/QNNs/qnn/qnlp/data/DataBase_docs - Hoja 1.csv', 7, 7)
circuits = c.create()
for i in circuits:
	print(circuits)
for _ in range(10):
	results = c.sample_run_global(np.random.normal(0, 2 * np.pi, 1000), 100)
	print(results)
print(c.params_used)
