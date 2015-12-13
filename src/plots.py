import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.interpolate import interp1d

def main():
	fig = plt.figure()
	gs = gridspec.GridSpec(1, 2)

	X = np.array([5, 20, 40, 95])
	ACC = [0.3714285714, 0.719047619, 0.8285714286, 0.8947368421]
	W = [10, 5, 2, 2]

	x_smooth = np.linspace(X.min(), X.max(), 200)
	acc_smooth = interp1d(X, ACC, kind='slinear')
	

	ax1 = fig.add_subplot(gs[0,0])
	ax1.plot(x_smooth, acc_smooth(x_smooth), 'b-')
	ax1.axis([0, 100, 0, 1])
	plt.ylabel('Test error')
	plt.xlabel('Dataset size')

	ax2 = fig.add_subplot(gs[0,1])
	ax2.plot(X, W, 'r-')
	ax2.axis([0, 100, 0, 10])
	plt. ylabel('Optimal window size')
	plt. xlabel('Dataset size')
	plt.show()

if __name__ == '__main__':
	main()