import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing

def main():
	data = io.load_data()
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y)

	analysis.plot_signals_two_column(data[class_names[y[0]]][0][:, 0:3],
									 X[0, 0:3, :].T,
										['Raw X', 'Raw Y', 'Raw Z'], 
										['Resampled X', 'Resampled Y', 'Resampled Z'])
	
	# analysis.plot_feature_space(preprocessing.flatten_matrix(X), y)

if __name__ == '__main__':
	main()