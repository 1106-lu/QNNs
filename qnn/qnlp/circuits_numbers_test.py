import numpy as np

from qnn.qnlp.circuits_words import CircuitsWords

parameters = np.random.normal(0, 2 * np.pi, 32)

c = CircuitsWords('C:/Users/usuario/Desktop/QIT/QNNs/qnn/qnlp/data/3Q DataBase.xlsx')
print(c)

cir, next = c.create('Nico have', 'cat')

# results = sample_run_global(c, parameters, 17)
# for i in results:
#	print(i)
