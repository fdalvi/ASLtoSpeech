import fnmatch
import numpy as np
import re
import os

def load_data(basepath="../data", quality="high"):
	""" 
	Reads the time series data from the given basepath. 

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
		datafiles = [os.path.join(dirpath, f) for dirpath, _, files in os.walk(os.path.join(basepath, 'high_quality')) for f in fnmatch.filter(files, '*.tsd')]

		for datafile in datafiles:
			# filename is of the form sign_name-1.tsd, where '1' is the instance number for that recording session
			filename = datafile.split('/')[-1];
			sign_name = filename.split('-')[0]
			sign_instance = filename.split('-')[1].split('.')[0]
			if sign_name not in data:
				data[sign_name] = []
			data[sign_name].append(np.genfromtxt(datafile, delimiter='\t'))
		return data
	elif quality=="low":
		# [X, Y, Z, roll, thumb bend, fore bend, index bend, ring bend]
		idx_array = [0,1,2,3,6,7,8,9]

		datafiles = [os.path.join(dirpath, f) for dirpath, _, files in os.walk(os.path.join(basepath, 'low_quality')) for f in fnmatch.filter(files, '*.sign')]
		for datafile in datafiles:
			# 	# filename is of the form alive0.tsd, where '0' is the instance number for that recording session
			match = re.search('.+\/(.+)(\d+)\.sign', datafile)
			sign_name = match.group(1)
			sign_instance = match.group(2)

			if 'cal-' in sign_name:
				continue
			if sign_name not in data:
				data[sign_name] = []
			time_series_data = np.genfromtxt(datafile, delimiter=',')
			if time_series_data.size == 0:
				continue
			time_series_data = time_series_data[:, idx_array]

			data[sign_name].append(time_series_data)
		return data
