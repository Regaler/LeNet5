import numpy as np
import mnist
import matplotlib.pyplot as plt
import util
import layer
import nn
import optimizer

"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""
#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

batch_size = 64
D_in = 784
D_out = 10

print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

### TWO LAYER NET FORWARD TEST ###
H=100
model = nn.TwoLayerNet(batch_size, D_in, H, D_out)

losses = []
optim = optimizer.SGD(model.get_params(), lr=0.0001)
#optim = optimizer.SGDMomentum(model.get_params(), lr=0.001, momentum=0.99)

for i in range(1500):
	X_batch, Y_batch = util.get_batch(X_train, Y_train, batch_size)
	Y_batch = util.MakeOneHot(Y_batch, D_out)

	Y_pred = model.forward(X_batch)

	loss = model.get_loss(Y_pred, Y_batch)
	print("epoch: %s, loss: %s" % (i, model.get_loss(Y_pred, Y_batch)))
	losses.append(loss)	

	model.backward()
	optim.step()

util.draw_losses(losses)