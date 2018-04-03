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
#H=400
#model = nn.TwoLayerNet(batch_size, D_in, H, D_out)
H1=300
H2=100
model = nn.ThreeLayerNet(batch_size, D_in, H1, H2, D_out)


losses = []
#optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
optim = optimizer.SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
criterion = loss.CrossEntropyLoss()

# TRAIN
ITER = 25000
for i in range(ITER):
	# get batch, make onehot
	X_batch, Y_batch = util.get_batch(X_train, Y_train, batch_size)
	Y_batch = util.MakeOneHot(Y_batch, D_out)

	# forward, loss, backward, step
	Y_pred = model.forward(X_batch)
	loss, dout = criterion.get(Y_pred, Y_batch)
	model.backward(dout)
	optim.step()
	
	if i % 100 == 0:
		print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
		losses.append(loss)
	

# save params
weights = model.get_params()
with open("weights.pkl","wb") as f:
	pickle.dump(weights, f)

util.draw_losses(losses)



# TRAIN SET ACC
Y_pred = model.forward(X_train)
result = np.argmax(Y_pred, axis=1) - Y_train
result = list(result)
print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

# TEST SET ACC
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))
"""


X = np.random.randn(7,3,4,4) # N, Cin, W, H
#conv = layer.Conv(3, 10, 3) # Cin, Cout, F
#Y = conv._forward(X)
#print(Y.shape)
#dX, dW, db = conv._backward(Y)
#print("dX.shape: " + str(dX.shape))
maxpool = layer.MaxPool(2,2)
Y = maxpool._forward(X)
print(X[0,0,:,:])
print(Y[0,0,:,:])