##
from qnn.embedding.generate_data import GenerateData
##
data_p = 20
dev = .2

gd_func = GenerateData(
	total_data_points=data_p,
	deviation=dev)

dict = gd_func.gen(data_p,dev)
##
gd_func.plot(dict)