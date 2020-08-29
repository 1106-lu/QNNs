import numpy as np
from qnn.embedding.circuits import Circuits


def cost(sample_vectors):
  A = sample_vectors[:int(len(sample_vectors)/2)]
  B = sample_vectors[int(len(sample_vectors)/2):]
  aa, bb, ab = [], [], []

  for i in range(len(A)):
    for a in range(len(A)):
      if not i >= a:
        aa.append((np.inner(A[i], A[a])**2))

  for i in range(len(B)):
    for a in range(len(B)):
      if not i >= a:
        #print(i, a)
        bb.append((np.inner(B[i], B[a])**2))

  for i in range(len(B)):
    for a in range(len(A)):
      ab.append((np.inner(A[i], B[a])**2))

  Dhs = 1 / 2 * (sum(aa) + sum(bb)) - sum(ab)
  cost = 1 - 0.5 * Dhs
  return cost

def g_parametershift(param, circuits, theta):
  cs_func = Circuits(depth=4)
  perturbation_vector = np.zeros(len(theta))
  perturbation_vector[param] = np.pi/4

  neg_vec = cs_func.sample(circuits, theta=(theta-perturbation_vector))
  pos_vec = cs_func.sample(circuits, theta=(theta+perturbation_vector))
  result = cost(pos_vec) - cost(neg_vec)
  return result

def g_finitedifference(param, circuits, params, epsilon):
  cs_func = Circuits(depth=4)
  perturbation_vector = np.zeros(len(params))
  perturbation_vector[param] = 1

  new_neg = params - epsilon*perturbation_vector
  new_pos = params + epsilon*perturbation_vector
  neg_vec = cs_func.sample(circuits, theta=new_neg)
  pos_vec = cs_func.sample(circuits, theta=new_pos)

  result = (cost(pos_vec) - cost(neg_vec))/2*epsilon
  return result



