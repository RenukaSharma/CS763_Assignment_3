import yaml

def read_yaml():
	with open("configs/config.yaml", 'r') as stream:
		try:
			return yaml.load(stream)
		except yaml.YAMLError as exc:
			return exc
#read yaml files that defines hyperparameters and the location of data