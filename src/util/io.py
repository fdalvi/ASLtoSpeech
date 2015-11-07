import fnmatch
import numpy as np
import os

def load_data(basepath="../data", quality="high"):
	data = {}
	if quality == "high":
		datafiles = [os.path.join(dirpath, f) for dirpath, _, files in os.walk(basepath) for f in fnmatch.filter(files, '*.tsd')]

		for datafile in datafiles:
			# filename is of the form sign_name-1.tsd, where '1' is the instance number for that recording session
			filename = datafile.split('/')[-1];
			sign_name = filename.split('-')[0]
			sign_instance = filename.split('-')[1].split('.')[0]
			if sign_name not in data:
				data[sign_name] = []
			data[sign_name].append(np.genfromtxt(datafile, delimiter='\t'))
		return data