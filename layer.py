import numpy as np
lr = 0.001

class FC():
	"""
	Fully connected layer
	"""
	def __init__(self, N, D_in, D_out):
		#print("Build FC")
		self.W = np.random.rand(D_in, D_out) - 0.5
		self.b = np.array(D_out) - 0.5
		self.cache = None

	def _forward(self, X):
		#print("FC: _forward")
		out = np.dot(X, self.W) + self.b
		self.cache = X
		return out

	def _backward(self, dout):
		#print("FC: _backward")
		X = self.cache
		dX = np.dot(dout, self.W.T).reshape(X.shape)
		dW = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
		db = np.sum(dout, axis=0)

		# Update the parameters
		self.W -= lr*dW
		self.b -= lr*db
		return dX

class ReLU():
	"""
	ReLU activation layer
	"""
	def __init__(self):
		#print("Build ReLU")
		self.cache = None

	def _forward(self, X):
		#print("ReLU: _forward")
		out = np.maximum(0, X)
		self.cache = X
		return out

	def _backward(self, dout):
		#print("ReLU: _backward")
		X = self.cache
		dX = np.array(dout, copy=True)
		dX[X <= 0] = 0
		return dX

class Softmax():
	"""
	Softmax activation layer
	"""
	def __init__(self):
		#print("Build Softmax")
		pass

	def _forward(self, X):
		#print("Softmax: _forward")
		maxes = np.amax(X, axis=1)
		maxes = maxes.reshape(maxes.shape[0], 1)
		e = np.exp(X - maxes)
		dist = e / np.sum(e, axis=1).reshape(e.shape[0], 1)
		return dist

	def _backward(self, dout):
		"""
		Regard Softmax and NLLLoss always come together
		"""
		#print("Softmax: _backward")
		return