import numpy as np

class dataloader():

	def __init__(self, train, test, labels, train_batch_size, test_batch_size):

		self.train = train
		self.test = test
		self.labels = labels

		self.r_s_t = np.arange(self.train.shape[0])
		np.random.shuffle(self.r_s_t)
		self.r_s_te = np.arange(self.test.shape[0])
		np.random.shuffle(self.r_s_te)
		
		self.current_train = 0
		self.current_test = 0
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size

	def getitem_train(self, index):

		if index >= self.len_train():
			np.random.shuffle(self.r_s_t)
			raise IndexError

		index = self.r_s_t[index*self.train_batch_size:min((index+1)*self.train_batch_size, self.train.shape[0])]
		data, target = self.train[self.r_s_t], self.labels[self.r_s_t]

		return data, target

	def getitem_test(self, index):

		if index >= self.len_train():
			np.random.shuffle(self.r_s_t)
			raise IndexError

		index = self.r_s_te[index*self.test_batch_size:min((index+1)*self.test_batch_size, self.test.shape[0])]
		data = self.test[self.r_s_t]

		return data

	def len_train(self):

		return self.train.shape[0]

	def len_test(self):

		return self.test.shape[0]

