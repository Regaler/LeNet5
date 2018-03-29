import numpy as np
import util
from layer import FC, ReLU, Softmax
import pickle

class Net():

	# Neural network super class
	def __init__():
		pass

	def forward():
		pass

	def backward():
		pass

	def get_params():
		pass

	def set_params():
		pass

class TwoLayerNet():
	
	#Simple 2 layer NN
	
	def __init__(self, N, D_in, H, D_out, weights=''):
		self.FC1 = FC(N, D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(N, H, D_out)

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
		dout = self.FC2._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.FC1._backward(dout)

	def get_params(self):
		return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

	def set_params(self, params):
		[self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params

class ThreeLayerNet():

	#Simple 3 layer NN
	
	def __init__(self, N, D_in, H1, H2, D_out, weights=''):
		self.FC1 = FC(N, D_in, H1)
		self.ReLU1 = ReLU()
		self.FC2 = FC(N, H1, H2)
		self.ReLU2 = ReLU()
		self.FC3 = FC(N, H2, D_out)

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
		a2 = self.ReLU2._forward(h2)
		h3 = self.FC3._forward(a2)
		return h3

	def backward(self, dout):
		dout = self.FC3._backward(dout)
		dout = self.ReLU2._backward(dout)
		dout = self.FC2._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.FC1._backward(dout)

	def get_params(self):
		return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

	def set_params(self, params):
		[self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params