import numpy as np

class FC():
	"""
	Fully connected layer
	"""
	def __init__(self, N, D_in, D_out):
		#print("Build FC")
		self.cache = None
		self.W = {'val': np.random.rand(D_in, D_out) - 0.5, 'grad': 0}
		self.b = {'val': np.array(D_out) - 0.5, 'grad': 0}

	def _forward(self, X):
		#print("FC: _forward")
		out = np.dot(X, self.W['val']) + self.b['val']
		self.cache = X
		return out

	def _backward(self, dout):
		#print("FC: _backward")
		X = self.cache
		dX = np.dot(dout, self.W['val'].T).reshape(X.shape)

		self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
		self.b['grad'] = np.sum(dout, axis=0)
		#self._update_params()
		return dX

	
	def _update_params(self, lr=0.001):
		# Update the parameters
		self.W['val'] -= lr*self.W['grad']
		self.b['val'] -= lr*self.b['grad']
	
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

class Sigmoid():
	"""
	Sigmoid activation layer
	"""
	def __init__(self):
		self.cache = None

	def _forward(self, X):
		self.cache = X
		return 1 / (1 + np.exp(-X))

	def _backward(self, dout):
		X = self.cache
		dX = dout*X*(1-X)
		return dX

class tanh():
	"""
	tanh activation layer
	"""
	def __init__(self):
		self.cache = X

	def _forward(self, X):
		self.cache = X
		return np.tanh(X)

	def _backward(self, X):
		X = self.cache
		dX = dout*(1 - np.tanh(X)**2)
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
		pass
