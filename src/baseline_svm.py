import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
from sklearn import svm
from sklearn import metrics

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

	analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, False)

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
		analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, True)

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
			analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, True)

	

if __name__ == '__main__':
	run_svm(quality="low", ablation=False)
