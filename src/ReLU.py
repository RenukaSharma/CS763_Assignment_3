import numpy as np

class ReLU():

	def __str__(self):

		return 'Acivation Function: ReLU'

	def forward(self, x):

		self.input = x

		x[x<0] = 0
		
		return x

	def clear_grad(self):

		return

	def __call__(self, x):

		return self.forward(x)

	def backward(self, gradOutput):

		gradInput = self.input.copy()
		gradInput[gradInput>0] = 1
		gradInput[gradInput<0] = 0
		
		return gradInput

# if __name__ == "__main__":

