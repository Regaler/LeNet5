import numpy as np
X = np.random.rand(4,4)
W,H = X.shape
F = 2
W_,H_ = 2,2

Y = np.zeros((W_,H_))
M = np.zeros(X.shape)
for w_ in range(W_):
    for h_ in range(H_):
        Y[w_,h_] = np.max(X[F*w_:F*(w_+1),F*h_:F*(h_+1)])
        i,j = np.unravel_index(X[F*w_:F*(w_+1),F*h_:F*(h_+1)].argmax(),(F,F))
        M[F*w_+i,F*h_+j] = 1
print(X)
print(Y)
print(M)
