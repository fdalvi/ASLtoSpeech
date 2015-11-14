import matplotlib.pyplot as plt
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.decomposition import RandomizedPCA

def plot_signals(signals, labels):
	"""
	Plots the various signals across time. All signals are
	required to have the same length (or number of samples).

	Args:
		signals: 2d-array, where each row is a measurement at
			an instance of time, and each column is a signal
		labels: list of labels for each signal (column) in the 
			signals matrix
	Returns
		None
	"""
	f, plots = plt.subplots(signals.shape[1], 1, sharex='col', sharey='row')

	for i in xrange(0, signals.shape[1]):
		plots[i].plot(xrange(1, signals[:,i].size+1), signals[:,i])

	plt.show()

def plot_signals_two_column(left_signals, right_signals, left_labels, right_labels):
	"""
	Plots the various signals across time in two columns. This is useful if one
	wants to compare signals or plot signals of different lengths.

	Args:
		left_signals: 2d-array, where each row is a measurement at
			an instance of time, and each column is a signal
		right_signals: 2d-array, where each row is a measurement at
			an instance of time, and each column is a signal
		left_labels: list of labels for each signal (column) in the 
			left_signals matrix
		right_labels: list of labels for each signal (column) in the 
			right_signals matrix
	Returns
		None
	"""
	subplot_rows = max([left_signals.shape[1], right_signals.shape[1]])
	subplot_columns = 2

	f, plots = plt.subplots(subplot_rows, subplot_columns, sharex='col', sharey='row')
	for i in xrange(0, left_signals.shape[1]):
		plots[i][0].plot(left_signals[:, i])
		plots[i][0].set_title(left_labels[i])

	for i in xrange(0, right_signals.shape[1]):
		plots[i][1].plot(right_signals[:, i])
		plots[i][1].set_title(right_labels[i])

	plt.show()

def plot_confusion_matrix(cm, class_names): 
	"""
	Plots a confusion matrix. 

	Args:
		cm: confusion matrix
		class_names: a list of class names where the index of the class
		name in the list corresponds to how it was labeled in the models

	Returns: 
		None
	"""
	##plotting unnormalized confusion matrix
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('confusion matrix of sign multiclassification')
	#plt.tight_layout()

	#uncomment for actual labels 
	# tick_marks = np.arange(len(class_names))
	# plt.xticks(tick_marks, class_names, rotation=90, fontsize=5)
	# plt.yticks(tick_marks, class_names, fontsize=5)

	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()

def output_error(y_predict, y_true): 
	"""
	Outputs several performance metrics of a given model, including precision, 
	recall, f1score, and error.

	Args:
		y_predict: an array of the predicted labels of the examples 
		y_true: an array of the true labels of the examples

	Returns
		(precision, recall, fscore, _), error 
	"""
	return metrics.precision_recall_fscore_support(y_true, y_predict), np.sum(y_predict != y_true) / float(y_predict.shape[0])

def reduce_dimensions(data, n, random_state=None):
	"""
	Reduces the input data's dimension to 'n'.

	Args:
		data: An M x N matrix, where M is the number of samples and N is the number
			of features. The dimensions will be reduced from N to n.
		n: The new number of dimensions
	Returns:
		data: An M x n reduced dimension matrix.
	"""
	pca = RandomizedPCA(n_components = n, random_state=random_state)
	return pca.fit_transform(data)

def plot_feature_space(data, labels=None):
	"""
	Given the data, plots the data as a 3d plot (by reducing the dimensions using PCA).
	If labels are passed in, each class is given a color in the 3d plot.

	Args:
		data: an M x N matrix, where M is the number of points
		labels: an M x 1 matrix, with the labels for each of the M points
	Returns:
		None.
	"""
	data = reduce_dimensions(data, 3)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if labels != None:
		classes = np.sort(np.unique(labels))
		colors = []
		for i in classes:
			new_color = "".join([random.choice(['0','1','2','3','4','5','6','7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']) for i in xrange(6)])
			colors.append('#' + new_color)
		color = [colors[i] for i in labels]
	else:
		color = "#0000FF"

	# 1 - good, 0 - bad
	# base_colors = ['r','g']
	# colors = [base_colors[l] for l in image_labels]

	ax.scatter(data[:,0], data[:,1], data[:,2], c=color)
	plt.show()