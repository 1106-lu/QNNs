import matplotlib.pyplot as plt
import numpy as np

from qnn.qnlp.circuits_numbers import sample_run_global, cc_1234567
from qnn.qnlp.optimization import cost_global, g_parameter_shift_global

epsilon = .0000001
cost_plot = []
epoch_plot = []
param_plot = []
result_global = []
expected_bits = [[0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0]]

c = cc_1234567()
for i in c:
    print('\n \n', i)

epoch = 70
lr = .001
parameters = np.random.normal(0, 2 * np.pi, 32)

for o in range(epoch):
    result_global = sample_run_global(c, parameters, 100)
    cost_plot.append(cost_global(result_global, expected_bits))
    param_plot.append(parameters)
    epoch_plot.append(o)

    if o%5 == 0:
        print('d u wanna procced?')
        input()
        if input()=='nope':
            min_i = 0
            min_cost = cost_plot[0]
            for i in range(len(cost_plot)):
                if cost_plot[i] < min_cost:
                    min_cost = cost_plot[i]
                    min_i = i
            print('min COST:', min_cost)
            print('min EPOCH:', min_i)
            print('min PARAM:', param_plot[min_i])
            exit()
        else:
            pass

    for i in range(len(parameters)):
        parameters[i] = parameters[i] - lr * (g_parameter_shift_global(c, i, parameters, expected_bits) + epsilon**2)
    if o != 0:
        if not lr == 0.1144754599728827:
            if (cost_plot[o]) < (cost_plot[o - 1]):
                lr = lr * 1.2
            else:
                lr = lr

    print('Epoch:', o, 'Cost:', cost_plot[o], 'LearningRate:', lr)

min_i = 0
min_cost = cost_plot[0]
for i in range(len(cost_plot)):
    if cost_plot[i] < min_cost:
        min_cost = cost_plot[i]
        min_i = i

print('min COST:', min_cost)
print('min EPOCH:', min_i)
print('min PARAM:', param_plot[min_i])

result = sample_run_global(c, param_plot[min_i], 1000)
for i in result:
    print(i, '\n')

plt.plot(epoch_plot, cost_plot)
plt.show()
print(param_plot[min_i])
