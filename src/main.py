import util.io as io
import util.preprocessing as preprocessing

def main():
	data = io.load_data()
	print data['alive'][1].shape
	print preprocessing.to_fixed_length(data['alive'], 57)[1].shape

if __name__ == '__main__':
	main()