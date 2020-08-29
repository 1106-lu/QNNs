import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GenerateData:

	def __init__(self,
	             total_data_points: int):
		self.total_data_points = total_data_points

	def gen(self, total_data_points):
		label_11 = []
		label_0 = []
		label_12 = []

		s_11 = np.random.default_rng().normal(-1.5, .2, total_data_points / 4)
		s_0 = np.random.default_rng().normal(0, .2, total_data_points / 2)
		s_12 = np.random.default_rng().normal(1.5, .2, total_data_points / 4)

		for i in range(len(s_11)):
			label_11.append(-1)
			label_12.append(-1)
		for i in range(len(s_0)):
			label_0.append(1)

		x = np.append(s_11, s_12)
		x = np.append(x, s_0)
		x_label = np.append(label_11, label_12)

		return pd.DataFrame({'DataPoint': x, 'Label': x_label}), x, x_label

	def plot(self, total_data_points):
		data_dict, x, x_label = self.gen(total_data_points)
		y = np.ones(total_data_points)
		area = 50

		plt.scatter(x, y, area, x_label, alpha=.5)
		return plt.show()