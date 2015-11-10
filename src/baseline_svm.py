import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
from sklearn import svm

def flatten_matrix(X): 
	"""
	Flattens a tensor matrix X to two dimensions. 

	Args:
		X: matrix with dimensions (x,y,z)

	Returns: 
		flattened matrix of (x,y*z)
	"""
	return X.swapaxes(1,2).reshape((X.shape[0], X.shape[1]*X.shape[2]))

def run_analyses(y_predict_train, y_train, y_predict, y_test, class_names): 
	# calculate metrics
	_, training_error = analysis.output_error(y_predict_train, y_train)
	(precision, recall, f1, _), testing_error = analysis.output_error(y_predict, y_test)
	class_names_list = [class_names[index] for index in class_names.keys()]
	analysis.plot_confusion_matrix(y_predict, y_test, class_names_list)

	# print out metrics
	print 'Average Precision:', np.average(precision)
	print 'Average Recall:', np.average(recall)
	print 'Average F1:', np.average(f1)
	print 'Training Error:', training_error
	print 'Testing Error:', testing_error

def run_svm(quality="high"):
	"""
	Runs a simple SVM model with a linear kernel; first fits the model
	on the training data (70 percent of the total data) and tests on 
	the rest of the data.

	Args:
		None

	Returns: 
		None
	"""
	data = io.load_data(quality=quality)
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=True)
	# flattened_Xtrain = flatten_matrix(np.hstack((X_train[:,0:3,:], X_train[:,11:14,:])))
	# flattened_Xtest = flatten_matrix(np.hstack((X_test[:,0:3,:], X_test[:,11:14,:])))	
	flattened_Xtrain = flatten_matrix(X_train)
	flattened_Xtest = flatten_matrix(X_test)	

	# fit svm model
	svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
	svm_model.fit(flattened_Xtrain, y_train)
	y_predict_train = svm_model.predict(flattened_Xtrain)
	y_predict = svm_model.predict(flattened_Xtest)

	run_analyses(y_predict_train, y_train, y_predict, y_test, class_names)

if __name__ == '__main__':
	run_svm()