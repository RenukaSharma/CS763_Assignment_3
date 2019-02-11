import numpy as np
from .Linear import Linear
from .ReLU import ReLU
from .Criterian import Criterian
from .Softmax import Softmax

class Model():

	def __init__(self):

		self.Layers = [
						Linear(),
						ReLU(),
						Linear(),
						Softmax(),
						]

		self.isTrain = True
		self.criterian = Criterian()

	def addLayer(self, layer):

		self.Layers.append(layer)

	def __call__(self, x):

		return self.forward(x)

	def forward(self, x):

		for i in self.Layers:
			x = i(x)

		return x

	def backward(self, x):

		for i in self.Layers:
			x = i.backward(x)

	def dispGradParam(self):

		for i in range(len(self.Layers)):

			print(self.Layers[len(self.Layers) - i - 1])

	def clearGradParam(self):

		for i in self.Layers:
			i.clear_grad()