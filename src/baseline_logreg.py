import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
from sklearn import linear_model
import baseline_svm

def run_logreg(quality='high'):
	"""
	Runs a simple logistic regression model; first fits the model
	on the training data (70 percent of the total data) and tests on 
	the rest of the data.

	Args:
		none

	Returns:
		none
	"""

	data = io.load_data(quality=quality)
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=True)

	# flatten data
	flattened_Xtrain = preprocessing.flatten_matrix(X_train) 
	flattened_Xtest = preprocessing.flatten_matrix(X_test)

	# fit logistic regression model
	logreg_model = linear_model.LogisticRegression(multi_class='ovr')
	logreg_model.fit(flattened_Xtrain, y_train)
	y_predict_train = logreg_model.predict(flattened_Xtrain)
	y_predict = logreg_model.predict(flattened_Xtest)

	# print metrics and confusion plot
	analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names)

if __name__ == '__main__':
	run_logreg('low')
