import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

from sknn.mlp import Classifier, Layer

def run_nn(quality='low'):
	"""
	Runs a simple neural network model; first fits the model
	on the training data (70 percent of the total data) and tests on 
	the rest of the data.

	Args:
		none

	Returns:
		none
	"""

	data = io.load_data(quality=quality)
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=False)

	y_train_one_hot = np.zeros((y_train.shape[0], len(class_names)))
	for i in range(y_train.shape[0]):
		y_train_one_hot[i, y_train[i]] = 1

	y_test_one_hot = np.zeros((y_test.shape[0], len(class_names)))
	for i in range(y_test.shape[0]):
		y_test_one_hot[i, y_test[i]] = 1

	# flip last two axes
	# tensor: samples x features x time
	# tensor: samples x time x features
	X_train = np.swapaxes(X_train, 1, 2)
	X_test = np.swapaxes(X_test, 1, 2)

	print X_train.shape
	print y_train_one_hot.shape

	lstm_model = Sequential()

	############################## MODEL 1 ##############################
	# Average Precision: 0.109463986213
	# Average Recall: 0.0912280701754
	# Average F1: 0.0664461045921
	# Training Error: 0.852201933405
	# Testing Error: 0.908771929825
	# HIDDEN_LAYER = 300
	# lstm_model.add(LSTM(output_dim=HIDDEN_LAYER, 
	# 					input_dim=X_train.shape[2], 
	# 					activation='tanh', 
	# 					inner_activation='hard_sigmoid',
	# 					return_sequences=False))
	# lstm_model.add(Dense(y_train_one_hot.shape[1], activation='tanh'))
	# lstm_model.compile(loss='mean_squared_error', optimizer='rmsprop')
	# lstm_model.fit(X_train, y_train_one_hot, batch_size=16, nb_epoch=10)
	############################## MODEL 1 ##############################
	#
	############################### MODEL 2 ##############################
	# Average Precision: 0.109463986213
	# Average Recall: 0.0912280701754
	# Average F1: 0.0664461045921
	# Training Error: 0.852201933405
	# Testing Error: 0.908771929825
	HIDDEN_LAYER = 300
	lstm_model.add(LSTM(output_dim=HIDDEN_LAYER, 
						input_dim=X_train.shape[2], 
						activation='tanh', 
						inner_activation='hard_sigmoid',
						return_sequences=False))
	
	lstm_model.add(LSTM(HIDDEN_LAYER,
						activation='tanh', 
						inner_activation='hard_sigmoid',
						return_sequences=False))
	lstm_model.add(Dense(y_train_one_hot.shape[1], activation='tanh'))
	lstm_model.compile(loss='mean_squared_error', optimizer='rmsprop')
	lstm_model.fit(X_train, y_train_one_hot, batch_size=16, nb_epoch=10)
	############################## MODEL 2 ##############################
	# lstm_model.add(Dropout(0.5))
	# lstm_model.add(Dense(95))
	# lstm_model.add(Activation('softmax'))

	
	# score = model.evaluate(X_test, Y_test, batch_size=16)

	y_predict_train = lstm_model.predict_classes(X_train)
	y_predict = lstm_model.predict_classes(X_test)
	# y_predict_one_hot = nn_model.predict(flattened_Xtest)

	# print y_predict
	# print y_predict_one_hot

	# print metrics and confusion plot
	analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names, False, False)

	print y_predict

if __name__ == '__main__':
	run_nn('low')
