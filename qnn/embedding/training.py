from qnn.embedding.optimization import cost, g_finitedifference#, g_parametershift
from qnn.embedding.circuits import Circuits

import time
import progressbar
import matplotlib.pyplot as plt

def train(circuits, lr, epsilon, epoch, params):
    #params = np.random.normal(0, 2*np.pi, 80)
    cs_func = Circuits(depth=4)
    sample_vectors = cs_func.sample(cs=circuits, theta=params)
    print(cost(sample_vectors))
    cost_plot, epoch_plot = [], []
    bar_func = progressbar.ProgressBar(maxval=epoch)

    for o in range(epoch):
      for i in range(len(params)):
        params[i] = params[i].real - lr*g_finitedifference(i, circuits, params, epsilon).real
        #params[i] = lr*params[i] + (1-lr)*(g_finitedifference(i, circuits, params, epsilon))**2
        #params[i] = lr*params[i] + (1-lr)*(g_parametershift(i, circuits, params))**2
        #print(params[i]-ols)
      sample_vectors = cs_func.sample(circuits, params)
      cost_plot.append(cost(sample_vectors).real)
      print('Epoch:', o+1, 'Cost:', cost_plot[o])
      epoch_plot.append(o)
      if o != 1 or 0:
          if cost_plot[o] < cost_plot[o-1]:
              lr = lr + lr*2
          else:
              lr = lr

    plt.plot(epoch_plot, cost_plot)
    plt.show()