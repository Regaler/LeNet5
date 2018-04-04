import numpy as np
import mnist
import matplotlib.pyplot as plt
import util
import layer
import nn
import optimizer
import pickle
import loss

"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)

#X_train = X_train[:6000]
#Y_train = Y_train[:6000]
#X_test = X_test[:1000]
#Y_test = Y_test[:1000]

batch_size = 16
D_out = 10

model = nn.LeNet5()
losses = []
optim = optimizer.SGD(model.get_params(), lr=0.00003)
#optim = optimizer.SGDMomentum(model.get_params(), lr=0.00003, momentum=0.80, reg=0.0003)
criterion = loss.SoftmaxLoss()

# Train
ITER = 30000
for i in range(ITER):
	# get batch, make onehot
	X_batch, Y_batch = util.get_batch(X_train, Y_train, batch_size)
	Y_batch = util.MakeOneHot(Y_batch, D_out)

	# forward, loss, backward, step
	Y_pred = model.forward(X_batch)
	loss, dout = criterion.get(Y_pred, Y_batch)
	model.backward(dout)
	optim.step()

	print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
	losses.append(loss)
	"""
	if i % 100 == 0:
		print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
		losses.append(loss)
	"""

# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

with open("losses.pkl","wb") as f:
	pickle.dump(losses, f)

#util.draw_losses(losses)

# Test
# TRAIN SET ACC
Y_pred = model.forward(X_train)
result = np.argmax(Y_pred, axis=1) - Y_train
result = list(result)
print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# TEST SET ACC
Y_pred = model.forward(X_test[:100])
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))