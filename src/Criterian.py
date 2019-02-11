import numpy as np


class Criterian():

	def __init__(self, loss='CEL', encoding='one_hot'):

		self.encoding = encoding

		self.loss = loss
		if loss == 'CEL':

			self.forward = self.CEL_loss

	def clear_grad(self):

		return

	def __str__(self):

		return 'Loss Function: CEL'

	def __call__(self, x):

		return self.forward(x)

	def CEL_loss(self, x, y):

		# x has the size of [batchsize, number of dims]
		# y has the size of [batchsize] if encoding = categorical else size = [batchsize, number of dims]


		if self.encoding == 'one_hot':

			y_ = y

		elif self.encoding == 'categorical':

			y_ = np.zeros([y.shape[0], x.shape[1]])
			y_[y] = 1

		return -np.mean(y_*np.log(x))