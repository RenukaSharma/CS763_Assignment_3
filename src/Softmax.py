import numpy as np

class Softmax():

	# def __init__(self):
	def __call__(self, x):

		return self.forward(x)

	def clear_grad(self):

		return

	def __str__(self):

		return 'Acivation Function: Softmax'

	def forward(self, x):

		# x has the shape [batchsize, number of dims]

		x1 = np.exp(x)

		return x1/np.sum(x1, axis=1)