import numpy as np
import src.nn as nn
from src.dataloader import dataloader
from src.read_yaml import read_yaml
from src.prepare_metadata import prepare_metadata
import torch
import csv

def train(dataloader, model, config):

	for i in range(dataloader.len_train()):

		data, target = dataloader.getitem_train(i)

		output = model(data)

		model.backward()

		if i!=0 and i%config['cal_accuracy'] == 0:
			if len(target.shape) == 1:
				print(np.mean((np.argmax(output, axis=1) == target).astype(np.float32)), ': Accuracy ', epoch, ': Epoch')
			else:
				print(np.mean((np.argmax(output, axis=1) == np.argmax(target, axis=1)).astype(np.float32)), ': Accuracy', epoch, ': Epoch')

def test(dataloader, model, config):

	predicted = []

	for i in range(dataloader.len_train()):

		data, target = dataloader.getitem_train(i)

		output = model(data)

		predicted += output.tolist()

	with open('output.csv', 'w') as f:

		wrtier = csv.writer(f)
		writer.write(predicted)
		
def seed(seed):

	np.random.seed(seed)

if __name__ == "__main__":

	config = read_yaml()
	seed(config['seed'])
	train_data, test_data, train_target = preapre_metadata()
	model = nn.Model()
	
	dataloader = dataloader(train_data, test_data, train_target, config['train_batch_size'], config['test_batch_size'])

	for epoch in range(config['epoch']):

		train(dataloader, model, config, epoch)
		test(dataloader, model, config, epoch)