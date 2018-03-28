import numpy as np
import mnist
import matplotlib.pyplot as plt
import util
import layer
import nn
import optimizer
import pickle

"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

batch_size = 32
D_in = 784
D_out = 10

print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

### TWO LAYER NET FORWARD TEST ###
H=300
model = nn.TwoLayerNet(batch_size, D_in, H, D_out)

losses = []
optim = optimizer.SGD(model.get_params(), lr=0.0001)
#optim = optimizer.SGDMomentum(model.get_params(), lr=0.001, momentum=0.99)

# TRAIN
for i in range(30000):
	X_batch, Y_batch = util.get_batch(X_train, Y_train, batch_size)
	Y_batch = util.MakeOneHot(Y_batch, D_out)

	Y_pred = model.forward(X_batch)

	loss, dout = model.get_loss(Y_pred, Y_batch)
	print("epoch: %s, loss: %s" % (i, loss))
	losses.append(loss)

	model.backward(dout)
	optim.step()

# TEST
Y_pred = model.forward(X_train)
#print(np.argmax(Y_pred, axis=1))
#print(Y_test)
print(Y_pred)

weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

util.draw_losses(losses)

print("Y_train[20]: " + str(Y_train[:20]))
print("Y_pred[20]" + str(np.argmax(Y_pred, axis=1)[:20]))

result = np.argmax(Y_pred, axis=1) - Y_train
result = list(result)
print("Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))
