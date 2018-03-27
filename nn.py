import numpy as np
import util
from layer import FC, ReLU, Softmax

class TwoLayerNet():
	"""
	Simple 2 layer NN
	"""
	def __init__(self, N, D_in, H, D_out):
		self.FC1 = FC(N, D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(N, H, D_out)
		self.softmax = Softmax()

		self.scores = np.array([])
		self.dout = None

	def forward(self, X):
		h1 = self.FC1._forward(X)
		a1 = self.ReLU1._forward(h1)
		h2 = self.FC2._forward(a1)
		a2 = self.softmax._forward(h2)
		self.scores = h2
		return a2

	def backward(self):
		self.get_dout()
		dout = self.FC2._backward(self.dout)
		dout = self.ReLU1._backward(dout)
		dout = self.FC1._backward(dout)

	def get_loss(self, Y_pred, Y_true):
		loss = util.NLLLoss(Y_pred, Y_true)
		self.loss = loss
		return loss

	def get_dout(self):
		self.dout = self.scores - 1
		return self.dout

	def get_params(self):
		return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]