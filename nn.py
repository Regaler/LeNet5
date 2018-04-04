import numpy as np
import util
from layer import FC, ReLU, Softmax, Dropout, Conv, MaxPool
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
		self.FC1 = FC(D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(H, D_out)

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
		self.FC1 = FC(D_in, H1)
		self.ReLU1 = ReLU()
		self.FC2 = FC(H1, H2)
		self.ReLU2 = ReLU()
		self.FC3 = FC(H2, D_out)

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


class LeNet5():
	# LeNet5

	def __init__(self):
		self.conv1 = Conv(1, 6, 5)
		self.ReLU1 = ReLU()
		self.pool1 = MaxPool(2,2)
		self.conv2 = Conv(6, 16, 5)
		self.ReLU2 = ReLU()
		self.pool2 = MaxPool(2,2)
		self.FC1 = FC(16*4*4, 120)
		self.ReLU3 = ReLU()
		self.FC2 = FC(120, 84)
		self.ReLU4 = ReLU()
		self.FC3 = FC(84, 10)
		self.Softmax = Softmax()

		self.p2_shape = None

	def forward(self, X):
		h1 = self.conv1._forward(X)
		a1 = self.ReLU1._forward(h1)
		p1 = self.pool1._forward(a1)
		h2 = self.conv2._forward(p1)
		a2 = self.ReLU2._forward(h2)
		p2 = self.pool2._forward(a2)
		self.p2_shape = p2.shape
		fl = p2.reshape(X.shape[0],-1) # Flatten
		h3 = self.FC1._forward(fl)
		a3 = self.ReLU3._forward(h3)
		h4 = self.FC2._forward(a3)
		a5 = self.ReLU4._forward(h4)
		h5 = self.FC3._forward(a5)
		a5 = self.Softmax._forward(h5)
		return a5

	def backward(self, dout):
		#dout = self.Softmax._backward(dout)
		dout = self.FC3._backward(dout)
		dout = self.ReLU4._backward(dout)
		dout = self.FC2._backward(dout)
		dout = self.ReLU3._backward(dout)
		dout = self.FC1._backward(dout)	
		dout = dout.reshape(self.p2_shape) # reshape
		dout = self.pool2._backward(dout)
		dout = self.ReLU2._backward(dout)
		dout = self.conv2._backward(dout)
		dout = self.pool1._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.conv1._backward(dout)

	def get_params(self):
		return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]