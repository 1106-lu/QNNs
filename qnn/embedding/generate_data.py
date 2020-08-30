import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GenerateDataRandomNormal():

	def __init__(self,
	             total_data_points: int,
	             deviation: float):
		self.total_data_points = total_data_points
		self.deviation = deviation

	def gen(self):
		label_11 = []
		label_0 = []
		label_12 = []

		s_11 = np.random.default_rng().normal(-1.5, self.deviation, int(self.total_data_points / 4))
		s_0 = np.random.default_rng().normal(0, self.deviation, int(self.total_data_points / 2))
		s_12 = np.random.default_rng().normal(1.5, self.deviation, int(self.total_data_points / 4))

		for i in range(len(s_11)):
			label_11.append(-1)
			label_12.append(-1)
		for i in range(len(s_0)):
			label_0.append(1)

		x = np.append(s_11, s_12)
		x = np.append(x, s_0)
		x_label = np.append(label_11, label_12)
		x_label = np.append(x_label, label_0)
		data_dict = pd.DataFrame({'DataPoint': x, 'Label': x_label}).transpose()

		return data_dict  # , x, x_label

	def plot(self, data_dict):
		y = np.ones(self.total_data_points)
		x = data_dict.values[0]
		x_label = data_dict.values[1]
		area = 50

		plt.scatter(x, y, area, x_label, alpha=.5)
		plt.show()
