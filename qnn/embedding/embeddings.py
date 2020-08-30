##
import numpy as np

from qnn.embedding.generate_data import GenerateData
from qnn.embedding.circuits import Circuits
from qnn.embedding.training import train
##
data_p = 20
dev = .2

gd_func = GenerateData(
	total_data_points=data_p,
	deviation=dev)

dict = gd_func.gen()
##
gd_func.plot(dict)

##
cs_func = Circuits(
	depth= 4)
cs = cs_func.create(dict.values[0])
theta_xd = np.random.normal(0, 2*np.pi, 80)
sample_vectors = cs_func.sample(cs, theta_xd)
#print(sample_vectors)

##
parameters = np.random.normal(0, 2*np.pi, 80)
train(cs, 500, .001, 10, parameters)