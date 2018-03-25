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

class Softmax():
	def __init__(self):
		print("Build Softmax")

	def _forward(self, X):
		maxes = np.amax(X, axis=1)
		maxes = maxes.reshape(maxes.shape[0], 1)
		e = np.exp(X - maxes)
		dist = e / np.sum(e, axis=1).reshape(e.shape[0], 1)
		return dist

	def _backward(self, dout, cache):
		pass

def NLLLoss(Y_pred, Y_true):
	"""
	
	"""
	loss = 0.0
	M = np.sum(Y_pred*Y_true, axis=1)
	for e in M:
		if e == 0:
			loss += 500
		else:
			loss += -np.log(e)
	return loss

def MakeOneHot(Y):
	Z = np.zeros((N, D_out))
	Z[np.arange(N), Y] = 1
	return Z

class TwoLayerNet():
	def __init__(self, D_in, H, D_out):
		self.FC1 = FC(N, D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(N, H, D_out)
		self.softmax = Softmax()

	def forward(self, X):
		h1 = self.FC1._forward(X)
		a1 = self.ReLU1._forward(h1)
		h2 = self.FC2._forward(a1)
		a2 = self.softmax._forward(h2)
		return a2

	def backward(self):
		pass

	def loss_function(self, Y_pred, Y_true):
		pass

# FC FORWARD TEST
"""
myFC = FC(N, D_in, D_out)
out = myFC._forward(X_train)
print(out)
"""

# TWO LAYER NET FORWARD TEST
H = 100
Y_train = MakeOneHot(Y_train)
model = TwoLayerNet(D_in, H, D_out)
Y_pred = model.forward(X_train)
#print(Y_pred)
loss = NLLLoss(Y_pred, Y_train)
print(loss)