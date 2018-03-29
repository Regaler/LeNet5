import numpy as np
import matplotlib.pyplot as plt
import random

def MakeOneHot(Y, D_out):
	N = Y.shape[0]
	Z = np.zeros((N, D_out))
	Z[np.arange(N), Y] = 1
	return Z

def draw_losses(losses):
	t = np.arange(len(losses))
	plt.plot(t, losses)
	plt.show()

def get_batch(X, Y, batch_size):
	N = len(X)
	i = random.randint(1, N-batch_size)
	return X[i:i+batch_size], Y[i:i+batch_size]