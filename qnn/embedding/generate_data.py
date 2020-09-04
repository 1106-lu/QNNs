import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GenerateRNBinaryDouble():

	def __init__(self,
	             total_data_points: int,
	             deviation: float):
		self.total_data_points = total_data_points
		self.deviation = deviation

	def gen(self):
		label_1 = []
		label_0 = []
		label_12 = []

		s_1 = np.random.default_rng().normal(-1.5, self.deviation, int(self.total_data_points / 4))
		s_0 = np.random.default_rng().normal(0, self.deviation, int(self.total_data_points / 2))
		s_12 = np.random.default_rng().normal(1.5, self.deviation, int(self.total_data_points / 4))

		for i in range(len(s_1)):
			label_1.append(-1)
			label_12.append(-1)
		for i in range(len(s_0)):
			label_0.append(1)

		x = np.append(s_1, s_12)
		x = np.append(x, s_0)
		x_label = np.append(label_1, label_12)
		x_label = np.append(x_label, label_0)
		data_dict = pd.DataFrame({'DataPoint': x, 'Label': x_label}).transpose()

		return data_dict, self.total_data_points

	def plot(self, data_dict):
		y = np.ones(self.total_data_points)
		x = data_dict.values[0]
		x_label = data_dict.values[1]
		area = 50

		plt.scatter(x, y, area, x_label, alpha=.5)
		plt.show()
		

class GenerateRNBinary():

	def __init__(self,
	             total_data_points: int,
	             deviation: float):
		self.total_data_points = total_data_points
		self.deviation = deviation

	def gen(self):
		label_1, label_0 = [], []

		s_1 = np.random.default_rng().normal(-1.5, self.deviation, int(self.total_data_points / 2))
		s_0 = np.random.default_rng().normal(0, self.deviation, int(self.total_data_points / 2))

		for i in range(len(s_1)):
			label_1.append(-1)
		for i in range(len(s_0)):
			label_0.append(1)

		x = s_1
		x = np.append(x, s_0)
		x_label = label_1
		x_label = np.append(x_label, label_0)
		data_dict = pd.DataFrame({'DataPoint': x, 'Label': x_label}).transpose()

		return data_dict, self.total_data_points

	def plot(self, data_dict):
		y = np.ones(self.total_data_points)
		x = data_dict.values[0]
		x_label = data_dict.values[1]
		area = 50

		plt.scatter(x, y, area, x_label, alpha=.5)
		plt.show()
