import numpy as np
import src.nn as nn
from dataloader import dataloader
from src.read_yaml import read_yaml

def train():

	pass

def test():

	pass

if __name__ == "__main__":

	model = nn.Model()
	config = read_yaml()
	train_dataloader = dataloader()