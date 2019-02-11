import numpy as np

class dataloader():

	def __init__(self, data, meta='', target='', type_='train'):

		self.type = type_
		self.data = data
		self.target = target
		self.meta = meta
		self.mean, self.stddev = np.load(self.meta)

	def __getitem__(self, index):

		if index >= self.__len__()
			raise IndexError

		data, target = self.load(index)
		data = (data - self.mean)/self.stddev

		return data, target

	def __len__(self):

		return len(self.paths)

