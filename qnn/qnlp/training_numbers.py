from qnn.qnlp.circuits_numbers import create_circuits, sample_run
from qnn.qnlp.optimization import cost, g_finite_difference
import numpy as np
import matplotlib.pyplot as plt


parameters = np.random.normal(0, 2 * np.pi, 33)

lr = 2
epsilon = .0001
cost_plot = []
epoch_plot = []

c = create_circuits()

epoch = 7

for o in range(epoch):
    result = sample_run(c, parameters, 100)
    cost_plot.append(cost(result, 1))
    epoch_plot.append(o)

    for i in range(len(parameters)):
        parameters[i] = parameters[i] - lr * g_finite_difference(c, i, parameters, epsilon)

    if o != 0:
        if (cost_plot[o]) < (cost_plot[o - 1]):
            lr = lr
        else:
            lr = lr / 2

    print('Epoch:', o, 'Cost:', cost_plot[o], 'LearningRate:', lr)

print(sample_run(c, parameters, 100))

plt.plot(epoch_plot, cost_plot)
plt.show()
