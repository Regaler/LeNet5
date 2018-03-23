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
		out = np.dot(X, self.W) + self.b
		return out

	def _backward(self, dout, cache):
		pass

class ReLU():
	def __init__(self, D_in, D_out):
		print("Build ReLU")
		pass

	def _forward(self, X):
		out = np.maximum(0, X)
		return out

	def _backward(self, dout, cache):
		pass

class TwoLayerNet():
	def __init__(self, D_in, H, D_out):
		self.FC1 = FC(D_in, H)
		self.FC2 = FC(H, D_out)

	def forward(self, X):
		self.FC1._forward(X)

	def backward(self):
		pass

	def loss_function(self):
		pass

myFC = FC(N, D_in, D_out)
out = myFC._forward(X_train)
print(out)





"""
model = LeNet5()
for epoch in range(100):
	Y_pred = model.forward(X_train)

	loss = model.loss_function(Y_pred, Y_train)
	
	model.backward(loss)

"""