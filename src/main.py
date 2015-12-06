import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing

def main():
	data = io.load_data(quality="low")
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y)
	X = preprocessing.scale_spatially(X)


	# for i in class_names:
	# 	# print i
	# 	# print class_names[i]
	# 	# print np.where(y==i)
	# 	# print np.where(y==i)[0]
	# 	# print np.where(y==i)[0].size
	# 	print class_names[i], i, np.where(y==i)[0].size

	# analysis.plot_signals_two_column(data[class_names[y[0]]][0][:, 0:3],
	# 								 X[0, 0:3, :].T,
	# 									['Raw X', 'Raw Y', 'Raw Z'], 
	# 									['Resampled X', 'Resampled Y', 'Resampled Z'])
	
	shop_idx = np.where(y==3)[0]
	shop_idx = shop_idx[0:6]

	# print X[shop_idx, 0, :].shape
	NUM = 3
	C1 = 0
	C2 = 3
	d1 = X[np.where(y==C1)[0][0:NUM], 3, :].T
	d2 = X[np.where(y==C2)[0][0:NUM], 3, :].T

	d1p = np.roll(d1, -1, 0) - d1
	# d1p = d1p[0:d1.shape[0]-1]
	d1p[-1, :] = d1p[-2, :]

	d2p = np.roll(d2, -1, 0) - d2
	d2p[-1, :] = d2p[-2, :]

	labels1 = [str(class_names[C1]) + ' ' + str(i) for i in xrange(NUM)] + [str(class_names[C1]) + '\' ' + str(i) for i in xrange(NUM)]
	labels2 = [str(class_names[C2]) + ' ' + str(i) for i in xrange(NUM)] + [str(class_names[C2]) + '\' ' + str(i) for i in xrange(NUM)]

	print d1.shape, d1p.shape
	print d2.shape, d2p.shape

	print np.concatenate((d1, d1p), 1).shape
	
	analysis.plot_signals_two_column(np.concatenate((d1, d1p), 1),
									 np.concatenate((d2, d2p), 1),
									labels1,
									labels2)

	# analysis.plot_signals_two_column(d1p,
	# 								 d2p,
	# 								labels1,
	# 								labels2)

if __name__ == '__main__':
	main()