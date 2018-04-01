import numpy as np

class FC():
	"""
	Fully connected layer
	"""
	def __init__(self, N, D_in, D_out):
		#print("Build FC")
		self.cache = None
		self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
		self.b = {'val': np.random.randn(D_out), 'grad': 0}

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

class Dropout():
	"""
	Dropout layer
	"""
	def __init__(self, p=1):
		self.cache = None
		self.p = p

	def _forward(self, X):
		M = (np.random.rand(*X.shape) < self.p) / self.p
		self.cache = X, M
		return X*M


	def _backward(self, dout):
		X, M = self.cache
		dX = dout*M/self.p
		return dX

class Conv():
	"""
	Conv layer
	"""
	def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
		self.Cin = Cin
		self.Cout = Cout
		self.F = F
		self.S = stride
		self.W = np.random.randn(Cout, Cin, F, F)
		self.b = np.random.randn(Cout)
		self.cache = None
		self.pad = padding

	def _forward(self, X):
		X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
		(N, Cin, H, W) = X.shape
		H_ = H - self.F + 1
		W_ = W - self.F + 1
		Y = np.zeros((N, self.Cout, H_, W_))

		for n in range(N):
			for c in range(self.Cout):
				for h in range(H_):
					for w in range(W_):
						Y[n, c, h, w] = np.sum(X[n, :, h:h+self.F, w:w+self.F] * self.W[c, :, :, :]) + self.b[c]

		self.cache = X
		return Y

	def _backward(self, dout):
		# dout (N,Cout,H_,W_)
		# W (Cout, Cin, F, F)
		X = self.cache
		(N, Cin, H, W) = X.shape
		H_ = H - self.F + 1
		W_ = W - self.F + 1
		W_rot = np.rot90(np.rot90(self.W))

		dX = np.zeros(X.shape)
		dW = np.zeros(self.W.shape)
		db = np.zeros(self.b.shape)

		# dW
		for co in range(self.Cout):
			for ci in range(Cin):
				for h in range(self.F):
					for w in range(self.F):
						dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])

		# db
		for co in range(self.Cout):
			db[co] = np.sum(dout[:,co,:,:])

		dout_pad = np.pad(dout, ((0,0),(0,0),(2,2),(2,2)), 'constant')
		print("dout_pad.shape: " + str(dout_pad.shape))
		# dX
		for n in range(N):
			for ci in range(Cin):
				for h in range(H):
					for w in range(W):
						dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

		return dX, dW, db