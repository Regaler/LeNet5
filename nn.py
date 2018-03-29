import numpy as np
import util
from layer import FC, ReLU, Softmax
import pickle

class TwoLayerNet():
	
	#Simple 2 layer NN
	
	def __init__(self, N, D_in, H, D_out, weights=''):
		self.N = N
		self.FC1 = FC(N, D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(N, H, D_out)
		self.softmax = Softmax()

		self.probs = np.array([])
		self.dout = None

		if weights == '':
			pass
		else:
			with open(weights,'rb') as f:
				params = pickle.load(f)
				self.set_params(params)

	def forward(self, X):
		h1 = self.FC1._forward(X)
		a1 = self.ReLU1._forward(h1)
		h2 = self.FC2._forward(a1)
		return h2

	def backward(self, dout):
		#print("dout is: " + str(self.dout))
		dout = self.FC2._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.FC1._backward(dout)

	def get_params(self):
		return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

	def set_params(self, params):
		[self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params
