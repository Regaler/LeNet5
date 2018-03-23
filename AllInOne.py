import numpy as np
import mnist

"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""
#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

N = 60
D_in = 784
D_out = 10
X_train, Y_train, X_test, Y_test = X_train[:60], Y_train[:60], X_test[:60], Y_test[:60]

print("N: " + str(N) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

class FC():
	def __init__(self, N, D_in, D_out):
		print("Build FC")
		self.W = np.random.rand(D_in, D_out) - 0.5
		self.b = np.array(D_out) - 0.5

	def _forward(self, X):
		print("FC: _forward")
		print(X.shape)
		print(self.W.shape)
		out = np.dot(X, self.W) + self.b
		cache = X
		return out

	def _backward(self, dout, cache):
		X = cache
		dX = np.dot(dout, self.W.T).reshape(x.shape)
		dW = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
		db = np.sum(dout, axis=0)
		return dX, dW, db

class ReLU():
	def __init__(self):
		print("Build ReLU")

	def _forward(self, X):
		out = np.maximum(0, X)
		cache = X
		return out

	def _backward(self, dout, cache):
		X = cache
		dX = np.array(dout, copy=True)
		dX[X <= 0] = 0
		return dX
		
def Softmax(Y_pred):
	"""Compute softmax values for each sets of scores in X."""
	maxes = np.amax(Y_pred, axis=1)
	maxes = maxes.reshape(maxes.shape[0], 1)
	e = np.exp(Y_pred - maxes)
	dist = e / np.sum(e, axis=1).reshape(e.shape[0], 1)
	#return np.sum(dist, axis=1)
	#return np.argmax(dist, axis=1)
	return dist

def CrossEntropyLoss(Y_pred, Y_true):
	"""
	
	"""
	loss = 0
	m = Y_pred.shape[0]
	loss = -(1.0/m) * np.sum(Y_true*np.log(Y_pred) + (1-Y_true)*np.log(1-Y_pred))
	return loss

class TwoLayerNet():
	def __init__(self, D_in, H, D_out):
		self.FC1 = FC(N, D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(N, H, D_out)

	def forward(self, X):
		h1 = self.FC1._forward(X)
		a1 = self.ReLU1._forward(h1)
		h2 = self.FC2._forward(a1)
		return h2

	def backward(self):
		pass

	def loss_function(self, Y_pred, Y_true):
		pass





















# LOSS FUNCTION TEST
"""
Y_pred = np.array([[0.1,0.8,0.1],[0.9,0.05,0.05]])
Y_true = np.array([[1,0,0],[0.5,0.5,0]])
#Y_pred = np.array([0.1,0.8,0.1])
#Y_true = np.array([0,1,0])
loss = CrossEntropyLoss(Y_pred,Y_true)
print(loss)
"""

# FC FORWARD TEST
"""
myFC = FC(N, D_in, D_out)
out = myFC._forward(X_train)
print(out)
"""

# TWO LAYER NET FORWARD TEST
"""
H = 100
model = TwoLayerNet(D_in, H, D_out)
out = model.forward(X_train)
A = Softmax(out)
print(A)
"""
