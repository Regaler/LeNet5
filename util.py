import numpy as np
import matplotlib.pyplot as plt

def NLLLoss(Y_pred, Y_true):
	"""
	Negative log likelihood loss
	"""
	loss = 0.0
	M = np.sum(Y_pred*Y_true, axis=1)
	for e in M:
		if e == 0:
			loss += 500
		else:
			loss += -np.log(e)
	return loss

def MakeOneHot(Y, D_out):
	N = Y.shape[0]
	Z = np.zeros((N, D_out))
	Z[np.arange(N), Y] = 1
	return Z

def draw_losses(losses):
	t = np.arange(len(losses))
	plt.plot(t, losses, 'r^')
	plt.show()