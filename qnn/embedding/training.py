from qnn.embedding.optimization import cost, g_parametershift, g_finitedifference
from qnn.embedding.circuits import Circuits

import numpy as np
import time
import progressbar
import matplotlib.pyplot as plt

def train(circuits, lr, epsilon, epoch, params):
  #epoch = 50
  #params = np.random.normal(0, 2*np.pi, 80)
  cs_func = Circuits(depth=4)
  sample_vectors = cs_func.sample(cs=circuits, theta=params)
  print(cost(sample_vectors))
  cost_plot = []
  epoch_plot = []
  bar =  progressbar.ProgressBar(maxval=epoch)
  bar.start()

  for o in range(epoch):
    for i in range(len(params)):
      params[i] = params[i] - lr*g_finitedifference(i, circuits, params, epsilon)
      #params[i] = lr*params[i] + (1-lr)*(g_finitedifference(i, circuits, params, epsilon))**2
      #params[i] = lr*params[i] + (1-lr)*(g_parametershift(i, circuits, params))**2
      #print(params[i]-ols)
      time.sleep(0.1)
      bar.update(o)
    sample_vectors = cs_func.sample(circuits, params)
    #print(sample_vectors)
    cost_plot.append(cost(sample_vectors).real)
    print(cost_plot[o])
    epoch_plot.append(o)

    if o != 1 or 0:
        if cost_plot[o] < cost_plot[o-1]:
          lr = lr + lr*2
          #print('yes')
        else:
          lr = lr

  plt.plot(epoch_plot, cost_plot)
  plt.show()