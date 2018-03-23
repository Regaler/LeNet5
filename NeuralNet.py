class LeNet5():
	"""

	"""
	def __init__(self):
		pass

	def forward(self):
		pass

	def backward(self):
		pass

	def train(self, train_config, data):
		return accuracy, losses

	def test(self, test_config, data):
		return accuracy, losses

def plot_losses(losses):
	pass


if __name__ == '__main__':
	LeNet5 = LeNet5()
	Dataloader = DataLoader('./MNIST', shuffle=True, train_ratio=0.7, transform=None, augmentation=None)
	train_config = {}
	test_config = {}

	accuracy_train, losses_train = LeNet5.train(train_config, Dataloader['Train'])
	print("Train accuracy is: " + str(accuracy))

	accuracy_test, losses_test = LeNet5.test(test_config, Dataloader['Test'])
	print("Test accuracy is: " + str(accuracy))