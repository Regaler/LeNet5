import numpy as np

class SGD():
	def __init__(self, params, lr=0.001):
		self.parameters = params
		self.lr = lr

	def step(self):
		for param in self.parameters:
			param['val'] -= self.lr*param['grad']

class SGDMomentum():
	def __init__(self, params, lr=0.001, momentum=0.99):
		self.l = len(params)
		self.parameters = params
		self.velocities = []
		for param in self.parameters:
			self.velocities.append(np.zeros(param['val'].shape))
		self.lr = lr
		#self.rho = momentum
		self.rho = 0.8

	def step(self):
		for i in range(self.l):
			self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
			self.parameters[i]['val'] -= self.lr*self.velocities[i]
