##
import numpy as np
from qnn.embedding.generate_data import GenerateRNBinary
from qnn.embedding.binary_double.training import Training

##
data_p = 20
dev = .2

gd_func = GenerateRNBinary(total_data_points=data_p, deviation=dev)
dict, data_num = gd_func.gen()
gd_func.plot(dict)
##
parameters = np.random.normal(0, 2*np.pi, 200)
##
tr = Training(data=dict.values[0],
              depth=4,
              learning_rate=500,
              epsilon=.001,
              epoch=10,
              initial_parameters=parameters)
tr.train
tr.minima()
