##
import numpy as np

from qnn.embedding.generate_data import GenerateDataRandomNormal
from qnn.embedding.circuits import Circuits
from qnn.embedding.training import Training

##
data_p = 20
dev = .2

gd_func = GenerateDataRandomNormal(
	total_data_points=data_p,
	deviation=dev)

dict = gd_func.gen()
##
gd_func.plot(dict)
##
cs_func = Circuits(
	depth=4)
cs = cs_func.create(dict.values[0])
##
parameters = np.random.normal(0, 2 * np.pi, 80)
tr = Training(circuits=cs,
              learning_rate=500,
              epsilon=.001,
              epoch=15,
              initial_parameters=parameters)
tr.train()
tr.minima()
