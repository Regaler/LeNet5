import numpy as np
from layer import Softmax

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
	return loss

class CrossEntropyLoss():
	def __init__(self):
		pass

	def get(self, Y_pred, Y_true):
		N = Y_pred.shape[0]
		softmax = Softmax()
		prob = softmax._forward(Y_pred)
		loss = NLLLoss(prob, Y_true)
		Y_serial = np.argmax(Y_true, axis=1)
		dout = prob.copy()
		dout[np.arange(N), Y_serial] -= 1
		return loss, dout