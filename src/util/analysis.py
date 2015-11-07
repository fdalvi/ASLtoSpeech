import matplotlib.pyplot as plt

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