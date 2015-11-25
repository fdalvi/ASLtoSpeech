import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
from sklearn import svm
from sklearn import metrics

def run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, ablation): 
	"""
	Runs analyses, including finding error, precision, recall, f1score, plotting
	a confusion matrix, on the results of a particular model. Prints out the numeric
	metrics and plots the graphical ones.

	Args:
		y_predict_train: 
			the predicted labels on the training examples
		y_train: 
			true labels on training examples
		y_predict: 
			predicted labels on testing examples
		y_test: 
			true labels on testing examples
		class_names: 
			dictionary that contains the class name that corresponds
			with the class index 

	Returns: 
		None
	"""
	# calculate metrics
	_, training_error = analysis.output_error(y_predict_train, y_train)
	(precision, recall, f1, _), testing_error = analysis.output_error(y_predict, y_test)
	class_names_list = [class_names[index] for index in class_names.keys()]
	if not ablation: 
		cm = metrics.confusion_matrix(y_test, y_predict)
		analysis.plot_confusion_matrix(cm, class_names_list)

	# print out metrics
	print 'Average Precision:', np.average(precision)
	print 'Average Recall:', np.average(recall)
	print 'Average F1:', np.average(f1)
	print 'Training Error:', training_error
	print 'Testing Error:', testing_error

def run_svm(quality="high", ablation=False):
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
	if ablation: 
		run_ablation_svm(X, y, class_names, quality)
		return 
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=False)	
	flattened_Xtrain = preprocessing.flatten_matrix(X_train)
	flattened_Xtest = preprocessing.flatten_matrix(X_test)	

	# fit svm model
	svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
	svm_model.fit(flattened_Xtrain, y_train)
	y_predict_train = svm_model.predict(flattened_Xtrain)
	y_predict = svm_model.predict(flattened_Xtest)

	run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, False)

def run_ablation_svm(X, y, class_names, quality):
	"""
	Runs ablation tests on svm model (i.e. runs the SVM on only a subset
	of features to determine which features are more important than others)

	Args:
		X: 
			input examples
		y: 
			labels of examples
		class_names: 
			list of actual signs
		quality: 
			flag that determines the quality of the data being passed in

	Returns: 
		None
	"""
	feature_array = preprocessing.get_feature_list(quality)
	feature_remove_list = []
	for feature in feature_array: 
		feature_remove_list.append(feature)
		ablated_X1 = preprocessing.get_ablated_matrix(X, quality, [feature])

		if len(feature_remove_list) > 1 and len(feature_remove_list) < len(feature_array): 
			ablated_X2 = preprocessing.get_ablated_matrix(X, quality, feature_remove_list)
		#create splits
		X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(ablated_X1, y, test_size=0.3, shuffle=False)	
		flattened_Xtrain = preprocessing.flatten_matrix(X_train)
		flattened_Xtest = preprocessing.flatten_matrix(X_test)	

		print "fitting svm on feature set without feature %s..." % feature 
		# fit svm model
		svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
		svm_model.fit(flattened_Xtrain, y_train)
		y_predict_train = svm_model.predict(flattened_Xtrain)
		y_predict = svm_model.predict(flattened_Xtest)
		
		print "running analyses on feature set without feature %s..." % feature
		run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, True)

		#true ablation tests
		if len(feature_remove_list) > 1 and len(feature_remove_list) < len(feature_array): 
			X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(ablated_X2, y, test_size=0.3, shuffle=False)	
			flattened_Xtrain = preprocessing.flatten_matrix(X_train)
			flattened_Xtest = preprocessing.flatten_matrix(X_test)	

			print "fitting svm on feature set without the features %s..." % ", ".join(feature_remove_list) 
			# fit svm model
			svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
			svm_model.fit(flattened_Xtrain, y_train)
			y_predict_train = svm_model.predict(flattened_Xtrain)
			y_predict = svm_model.predict(flattened_Xtest)
		
			print "running analyses on feature set without the features %s..." % ", ".join(feature_remove_list)
			run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, True)

	

if __name__ == '__main__':
	run_svm(quality="low", ablation=True)
