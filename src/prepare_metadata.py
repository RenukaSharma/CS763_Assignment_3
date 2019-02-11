import numpy as np
from torch.utils.serialization import load_lua

import matplotlib.pyplot as plt

class prepare_metadata():

	def __init__(self, path, meta):

		x = load_lua(path).numpy()
		average = np.mean(x, axis=0)
		x = x - average
		std_dev = np.sqrt(np.mean(np.square(x), axis=0))

		np.save(meta+'/average.npy', average)
		np.save(meta+'/std_dev.npy', std_dev)

if __name__ == "__main__":

	prepare_metadata('/home/mayank/Desktop/GitRepos/CS763_Assignment_3/dataset/data.bin', '/home/mayank/Desktop/GitRepos/CS763_Assignment_3/dataset/meta')