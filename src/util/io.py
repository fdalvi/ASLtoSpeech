import fnmatch
import numpy as np
import os

def load_data(basepath="../data", quality="high"):
	""" Reads the time series data from the given basepath. 

	Args:
		basepath: Root path where the data exists
			default: "../data"
		quality: The quality of data that should be returned.
			default: "high"
			Valid values: "high", "low"
	Returns:
		data: A dictionary where the key is a sign, and the value is a list
			of numpy matrices, each representing time series data for a specific
			instance of the signself.
	"""
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