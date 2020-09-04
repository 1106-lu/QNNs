import numpy as np

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
        bb.append((np.inner(B[i], B[a])**2))

  for i in range(len(B)):
    for a in range(len(A)):
      ab.append((np.inner(A[i], B[a])**2))

  Dhs = 1 / 2 * (sum(aa) + sum(bb)) - sum(ab)
  cost = 1 - 0.5 * Dhs
  return cost


