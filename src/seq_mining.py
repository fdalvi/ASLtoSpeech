import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing

trends = ['D', 'S', 'I']

def get_trend(idx):
	return trends[idx]

def get_trend_idx(trend):
	if trend == 'D':
		return 0
	if trend == 'S':
		return 1
	if trend == 'I':
		return 2
	return -1

def seg_mining():
	# Constants
	STEADY_THRESHOLD = 0.1

	# Loading data
	data = io.load_data(quality="low")
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X = preprocessing.scale_spatially(X)

	# Computing fake slopes
	dX = np.roll(X, -1, 2) - X
	dX[:, :, -1] = dX[:, :, -2]

	# Computing trends
	I_idx = np.where(dX > STEADY_THRESHOLD)
	D_idx = np.where(dX < -1*STEADY_THRESHOLD)
	S_idx = np.where(np.abs(dX) <= STEADY_THRESHOLD)
	trends = np.zeros(dX.shape, dtype=np.int8)
	trends[I_idx] = get_trend_idx('I')
	trends[S_idx] = get_trend_idx('S')
	trends[D_idx] = get_trend_idx('D')

	# print dX.shape, np.prod(dX.shape)
	# print I_idx[0].shape, S_idx[0].shape, D_idx[0].shape
	# print np.where(dX > STEADY_THRESHOLD)
	# print np.where(trends == 0)[0].shape, np.where(trends == 1)[0].shape, np.where(trends == 2)[0].shape
	
	# Combine trends
	# Intervals are inclusive in nature
	combined_trends = np.ones(trends.shape, dtype=np.int8) * get_trend_idx('useless')
	combined_trends_interval_start = np.ones(trends.shape, dtype=np.int8) * get_trend_idx('useless')
	combined_trends_interval_end = np.ones(trends.shape, dtype=np.int8) * get_trend_idx('useless')

	for e in xrange(combined_trends.shape[0]):
		for f in xrange(combined_trends.shape[1]):
			combined_trend_idx = 0
			combined_trends[e,f,0] = trends[e,f,0]
			combined_trends_interval_start[e,f,0] = 0
			for t in xrange(1, combined_trends.shape[2]):
				if trends[e,f,t] != trends[e,f,t-1]:
					# Set End time for previous trend
					combined_trends_interval_end[e,f,combined_trend_idx] = t-1

					# Start next trend
					combined_trend_idx += 1
					combined_trends[e,f,combined_trend_idx] = trends[e,f,t]
					combined_trends_interval_start[e,f,combined_trend_idx] = t

	# print trends[5, 3, :]
	# print combined_trends[5, 3, :]
	# print combined_trends_interval_start[5, 3, :]
	# print combined_trends_interval_end[5, 3, :]

if __name__ == '__main__':
	seg_mining()