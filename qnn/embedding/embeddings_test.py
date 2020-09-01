##
import numpy as np

from qnn.embedding.circuits import create
from qnn.embedding.generate_data import GenerateDataRandomNormal
from qnn.embedding.training import Training

##
data_p = 20
dev = .2

gd_func = GenerateDataRandomNormal(total_data_points=data_p, deviation=dev)
dict, data_num = gd_func.gen()
gd_func.plot(dict)
##
cs = create(data_x=dict.values[0], depth=4)
##
parameters = np.random.normal(0, 2*np.pi, 80)

tr = Training(circuits=cs,
              learning_rate=500,
              epsilon=.001,
              epoch=12,
              initial_parameters=parameters)
tr.train
tr.minima()