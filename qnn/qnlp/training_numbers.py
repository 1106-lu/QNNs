from qnn.qnlp.circuits_numbers import create_circuits, sample_run
from qnn.qnlp.optimization import cost, g_finite_difference
import numpy as np
import matplotlib.pyplot as plt


parameters = [ 0.34274625, -6.13012114,  2.95226212, -3.64527328,  0.5078258,  10.05113626,
            5.55305942,  0.34034723,  0.91667095, -8.55781144,  0.71497182, -1.22726473,
            0.39999794, -1.65117505,  0.78884599,  5.71482254,  4.25892603,  0.69167276,
            -2.37682877, -3.01871726,  2.14627255,  2.56209965,  3.58609136,  2.29447148,
            -3.35495532,  3.72053383,  7.89692506, -4.32840483,  9.7423093,   0.18789888,
            0.22549279,  3.39453389, 10.80278738]


epsilon = .0001
cost_plot = []
epoch_plot = []
learning_rate = [.001, .01, 1, 1.5, 2, 3, 4, 5]

c = create_circuits()

epoch = 7
for lr in learning_rate:
    print('Initial LearningRate:', lr)

    parameters = [ -2.46362637,  -1.23038732,   6.91978619,  -0.04303447,  -6.89783041,
                    6.52960166,  -6.45201144,   5.53443926,   6.5335287  ,  0.93038961,
                0.20196756, -11.13690625,  -4.25507322,   7.56943585 ,-4.23914388,
   4.81290731,   3.19158122,   14.9439866,  -1.38458494 ,  2.53127843,
  -2.20992347,  -1.43886095,   2.57794874,  -4.63522187 ,  5.49662408,
  -0.23046262,  -8.62185737, -10.93330163,   6.38161825 ,  3.08728987,
  -6.43763946,  -0.01967393,   1.15435534]

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
    print('############################')
