import numpy as np
import matplotlib.pyplot as plt
import random

def NLLLoss(Y_pred, Y_true):
	"""
	Negative log likelihood loss
	"""
	loss = 0.0
	N = Y_pred.shape[0]
	M = np.sum(Y_pred*Y_true, axis=1)
	for e in M:
		if e == 0:
			loss += 500
		else:
			loss += -np.log(e)
	return loss/N

def MakeOneHot(Y, D_out):
	N = Y.shape[0]
	Z = np.zeros((N, D_out))
	Z[np.arange(N), Y] = 1
	return Z

def draw_losses(losses):
	t = np.arange(len(losses))
	plt.plot(t, losses, 'r^')
	plt.show()

def CrossEntropyLoss(Y_pred, Y_true):
	m = Y_pred.shape[0]
	# cost = -(1.0/m) * (np.dot(np.log(Y_pred), Y_true.T) + np.dot(np.log(1-Y_pred), (1-Y_true).T))
	# cost = -(1.0/m) * np.sum(Y_true*np.log(Y_pred) + (1-Y_true)*np.log(1-Y_pred))
	return cost

def get_batch(X, Y, batch_size):
	N = len(X)
	i = random.randint(1, N-batch_size)
	return X[i:i+batch_size], Y[i:i+batch_size]