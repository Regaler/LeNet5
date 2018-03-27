import numpy as np
import mnist
import matplotlib.pyplot as plt
import util
import layer
import nn

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
Y_train = util.MakeOneHot(Y_train, D_out)
lr = 0.001

print("N: " + str(N) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))




### TWO LAYER NET FORWARD TEST ###
H=100
model = nn.TwoLayerNet(N, D_in, H, D_out)

losses = []
for i in range(1000):
	Y_pred = model.forward(X_train)

	loss = model.get_loss(Y_pred, Y_train)
	print("epoch: %s, loss: %s" % (i, model.get_loss(Y_pred, Y_train)))
	losses.append(loss)

	model.backward()

util.draw_losses(losses)