import numpy as np

class Linear():

	def __init__(self, in_dim, out_dim, initialisation='gaussian'):

		self.in_dim = in_dim
		self.out_dim = out_dim
		if initialisation == 'gaussian':
			self.initialise_g()
		# elif initialisation == 'xavier':
		# 	self.initialise_x()

		self.clear_grad()

	def clear_grad(self):

		self.gradW = np.zeros_like(self.W)
		self.gradB = np.zeros_like(self.B)
		self.gradInput = np.zeros([self.in_dim])

	def __str__(self):

		to_print = 'Hidden Layer: Linear\n, Weights: \n'+str(self.W)+'\nBias: \n'+str(self.B)
		return to_print

	def __call__(self, x):

		return self.forward(x)

	def initialise_g(self):

		self.W = np.random.normal(size=self.in_dim*self.out_dim).reshape([self.in_dim, self.out_dim])
		self.B = np.zeros([self.out_dim])

	def forward(self, x):

		self.input = x
		self.output = np.matmul(x, self.W) + self.B

		return self.output

	def backward(self, gradOutput): # input,  ToDo - IS this really required?

		self.gradB = gradOutput
		self.gradW = self.input*gradOutput
		self.gradInput = self.W*gradOutput

		return self.gradInput