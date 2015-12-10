import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
import re
import sys
import time
from sklearn import svm

from collections import Counter
# from sklearn import feature_selection.chi2

##decreasing: -0.3 < x <= 0
##very decreasing: -0.6 <= x <= -0.3 
##very very decreasing: -0.6 >= x

# Constants
STEADY_THRESHOLD = 0.1
INCREASING_THRESHOLD = 0.3 
VERY_INCREASING_THRESHOLD = 0.6 
NUM_SIGNS = 5
NUM_SIGNALS = 3
EXAMPLES_PER_SIGN = 70
NUM_PATTERN_FEATURES = 2000

trends = ['WD', 'VD', 'D', 'S', 'I', 'VI', 'WI']

def get_trend(idx):
	return trends[idx]

def get_trend_idx(trend):
	if trend in trends: 
		return trends.index(trend)
	return -1 
	# if trend == 'D':
	# 	return 0
	# if trend == 'S':
	# 	return 1
	# if trend == 'I':
	# 	return 2
	# return -1

def generate_one_pattern(D):
	patterns = dict()

	##instead of iterating through all signals, go through only 3
	idx = 0
	for e in xrange(D.shape[0]):
		for s in xrange(D.shape[1]):
			for t in xrange(D.shape[2]):
				if D[e,s,t] == -1:
					break
				patterns[get_trend(D[e,s,t]) + ':' + str(s+1)] = idx
				idx += 1

	return patterns

def generate_k_patterns(patterns, k, total_time):
	new_patterns = dict()
	if k == 2:
		for pattern_1 in patterns:
			for pattern_2 in patterns:
				if pattern_1 != pattern_2:
					signal_1 = pattern_1[2]
					signal_2 = pattern_2[2]

					if signal_1 != signal_2:
						# Generate 'overlap'
						new_patterns[pattern_1 + '-o;' + pattern_2] = patterns[pattern_1]
						new_patterns[pattern_2 + '-o;' + pattern_1] = patterns[pattern_2]

					new_patterns[pattern_1 + '-b;' + pattern_2] = patterns[pattern_1]
					new_patterns[pattern_2 + '-b;' + pattern_1] = patterns[pattern_2]
	else:
		for pattern_1 in patterns:
			for pattern_2 in patterns:
				if pattern_1 != pattern_2:
					# start_time = time.time()
					
					# sep_1_idx = pattern_1.rfind(';')
					# sep_2_idx = pattern_2.rfind(';')
					# prefix_1 = pattern_1[: sep_1_idx-2]
					# prefix_2 = pattern_2[: sep_2_idx-2]

					# if prefix_1 != prefix_2:
					# 	end_time = time.time()
					# 	total_time += (end_time-start_time)
					# 	continue
					#
					start_time = time.time() 	
					# sep_1_idx = pattern_1.rfind(';')
					# sep_2_idx = pattern_2.rfind(';')

					sep_1_idx = len(pattern_1)-4
					sep_2_idx = len(pattern_2)-4

					# print sep_1_idx, len(pattern_1)-4
					# print sep_2_idx, len(pattern_2)-4

					if patterns[pattern_1] != patterns[pattern_2]:
						end_time = time.time()
						total_time += (end_time-start_time)
						continue

					end_time = time.time()
					total_time += (end_time-start_time)

					suffix_1 = pattern_1[sep_1_idx+1:]
					suffix_2 = pattern_2[sep_2_idx+1:]

					signal_1 = pattern_1[sep_1_idx-3]
					signal_2 = pattern_2[sep_2_idx-3]
					relation_1 = pattern_1[sep_1_idx-1]
					relation_2 = pattern_2[sep_2_idx-1]

					if signal_1 != signal_2:
						# Generate 'overlap'
						new_patterns[pattern_1 + '2-o;' + suffix_2] = patterns[pattern_1]
						if not (relation_1 == 'o' and relation_2 == 'b'):
							new_patterns[pattern_2 + '-o;' + suffix_1] = patterns[pattern_2]

					new_patterns[pattern_1 + '-b;' + suffix_2] = patterns[pattern_1]
					if not (relation_1 == 'o' and relation_2 == 'b'):
						new_patterns[pattern_2 + '-b;' + suffix_1] = patterns[pattern_2]

	return new_patterns, total_time

def pattern_exists_recurse(D, states, relations, state_idx, relation_idx, w, cur_window, prev_interval):
	##base case 
	if state_idx == len(states):
		return True
	

	# if state_idx + 1 != len(states):
	# 	print states[state_idx+1]	
	T, start_intervals, end_intervals = D

	current_state = states[state_idx]
	signal = int(current_state[2]) - 1
	trend = get_trend_idx(current_state[0])

	match_idx = np.where(trend == T[signal, :])[0]
	if cur_window != None: 
		window_start, window_end = cur_window

	##run through all matches in T 
	for match in match_idx:
		start, end = start_intervals[signal, match], end_intervals[signal, match]

		##update window and check if it's within window size
		if cur_window == None: 
			window_start = end
		else: 
			if end < window_start: 
				window_start = end

		window_end = start 	
		if window_end - window_start > w: 
			return False 

		##check if pattern exists
		if prev_interval != None:
			#print states[state_idx], relations[relation_idx]
			prev_start, prev_end = prev_interval
			#print "(%d,%d) -> (%d,%d)"%(prev_start, prev_end, start, end)

			# check b
			if relations[relation_idx] == 'b':
				if prev_end > start:
					return False
			else:
				# check o
				if prev_end < start or start < prev_start:
					return False
	
		##recursive case

		return pattern_exists_recurse(D, states, relations, state_idx+1, relation_idx+1, w, (window_start, window_end),(start, end))

	return False

def pattern_exists(D, P, w):
	states = [P[m.start()-1:m.start()+2] for m in re.finditer(':', P)]
	relations = [P[m.start()+1] for m in re.finditer('-', P)]

	if pattern_exists_recurse(D, states, relations, 0, -1, w, None, None):
		return 1
	return 0

def support(D, P, w, minsup):
	current_sum = 0
	for i in xrange(D[0].shape[0]): 
		current_sum += pattern_exists((D[0][i,:,:], D[1][i,:,:], D[2][i,:,:]), P, w) 
		# if current_sum >= minsup: 
		# 	break

	return current_sum 	
	# return sum([pattern_exists((D[0][i,:,:], D[1][i,:,:], D[2][i,:,:]), P, w) for i in xrange(D[0].shape[0])])

def prune_patterns(D, patterns, minsup, w=20):
	pruned_patterns = dict()

	supports = dict()
	for i, pattern in enumerate(patterns):
		if i % int(0.05*len(patterns)) == 0:
			sys.stdout.write('=')
			sys.stdout.flush()

		pattern_support = support(D, pattern, w, minsup)
		if pattern_support >= minsup:
			# print pattern, patterns[pattern]
			pruned_patterns[pattern] = patterns[pattern]
			supports[pattern] = pattern_support
	print ''

	return pruned_patterns, supports


def chi_square(patterns, pattern_counts, all_pattern_supports): 
	##TODO: Convert to floats!!!!
	chi_square_statistics = Counter()
	num_instances = EXAMPLES_PER_SIGN * NUM_SIGNS
	print patterns
	for i in xrange(len(patterns)): 
		for iteration_number, pattern_dict in enumerate(patterns[i]):
			if all_pattern_supports[i][iteration_number] is None:
				continue
			for pattern in pattern_dict:
				p_divide_n = pattern_counts[pattern] / float(num_instances) 
				c_divide_n = NUM_SIGNS / float(num_instances)

				term_1 = ((all_pattern_supports[i][iteration_number][pattern] - p_divide_n * c_divide_n) ** 2) / (p_divide_n * c_divide_n)
				term_2 = ((EXAMPLES_PER_SIGN - all_pattern_supports[i][iteration_number][pattern]) / num_instances - ((1 - p_divide_n) * c_divide_n)) ** 2 \
									/ ((1 - p_divide_n) * c_divide_n)
				chi_square_statistics[pattern] += term_1 + term_2

	##use statistics to frank patterns 
	chi_square_statistics = [key for key, _ in sorted(chi_square_statistics.iteritems(), key=lambda (k,v): (v,k))]

	return chi_square_statistics


def construct_feature_vectors(ranked_patterns, D, w=20): 
	new_feature_matrix = np.zeros((EXAMPLES_PER_SIGN * NUM_SIGNS, len(ranked_patterns)))

	for i in xrange(new_feature_matrix.shape[0]): 
		new_feature_matrix[i,:] = np.array([pattern_exists((D[0][i,:,:], D[1][i,:,:], D[2][i,:,:]), P, w) for P in ranked_patterns])

	return new_feature_matrix

def seg_mining(use_all_signs):
	# Loading data
	data = io.load_data(quality="low")
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=False)

	if not use_all_signs: 
		# TODO: change not break
		X_train = preprocessing.scale_spatially(X_train)[:NUM_SIGNS * EXAMPLES_PER_SIGN,:NUM_SIGNALS,:]
		y_train = y_train[:NUM_SIGNS * EXAMPLES_PER_SIGN]
		X_test = preprocessing.scale_spatially(X_test)[:NUM_SIGNS * EXAMPLES_PER_SIGN,:NUM_SIGNALS,:]
		y_test = y_test[:NUM_SIGNS * EXAMPLES_PER_SIGN]
	else:  
		X_train = preprocessing.scale_spatially(X_train)
		X_test = preprocessing.scale_spatially(X_test)

	# print X.shape 
	# sys.exit()
	# Computing fake slopes
	dX = np.roll(X, -1, 2) - X
	dX[:, :, -1] = dX[:, :, -2]

	# Computing trends
	# binary_I_idx = (INCREASING_THRESHOLD >= dX) & (dX > STEADY_THRESHOLD)
	# I_idx = np.where(binary_I_idx == 1)
	# binary_VI_idx = (VERY_INCREASING_THRESHOLD >= dX) & (dX > INCREASING_THRESHOLD)
	# VI_idx = np.where(binary_VI_idx == 1)
	# WI_idx = np.where(dX > VERY_INCREASING_THRESHOLD)
	# binary_D_idx = (-1*INCREASING_THRESHOLD <= dX) & (dX < -1*STEADY_THRESHOLD)
	# D_idx = np.where(binary_D_idx == 1)
	# binary_VD_idx = (-1*VERY_INCREASING_THRESHOLD <= dX) & (dX < -1*INCREASING_THRESHOLD)
	# VD_idx = np.where(binary_VD_idx == 1)
	# WD_idx = np.where(dX < -1*VERY_INCREASING_THRESHOLD)
	# S_idx = np.where(np.abs(dX) <= STEADY_THRESHOLD)
	
	# trends = np.zeros(dX.shape, dtype=np.int8)
	# trends[I_idx] = get_trend_idx('I')
	# trends[VI_idx] = get_trend_idx('VI')
	# trends[WI_idx] = get_trend_idx('WI')
	# trends[S_idx] = get_trend_idx('S')
	# trends[D_idx] = get_trend_idx('D')
	# trends[VD_idx] = get_trend_idx('VD')
	# trends[WD_idx] = get_trend_idx('WD')
	
	I_idx = np.where(dX > STEADY_THRESHOLD)
	S_idx = np.where(np.abs(dX) <= STEADY_THRESHOLD)
	D_idx = np.where(dX < -1*STEADY_THRESHOLD)
	
	trends = np.zeros(dX.shape, dtype=np.int8)
	trends[I_idx] = get_trend_idx('I')
	trends[S_idx] = get_trend_idx('S')
	trends[D_idx] = get_trend_idx('D')

	# print dX.shape, np.prod(dX.shape)
	# print I_idx[0].shape, S_idx[0].shape, D_idx[0].shape
	# print np.where(dX > STEADY_THRESHOLD)
	# print np.where(trends == 0)[0].shape, np.where(trends == 1)[0].shape, np.where(trends == 2)[0].shape
	
	# Combine trends
	# Intervals are inclusive in nature
	combined_trends = np.ones(trends.shape, dtype=np.int8) * get_trend_idx('useless')
	combined_trends_interval_start = np.ones(trends.shape, dtype=np.int8) * get_trend_idx('useless')
	combined_trends_interval_end = np.ones(trends.shape, dtype=np.int8) * get_trend_idx('useless')

	for e in xrange(combined_trends.shape[0]):
		for f in xrange(combined_trends.shape[1]):
			combined_trend_idx = 0
			combined_trends[e,f,0] = trends[e,f,0]
			combined_trends_interval_start[e,f,0] = 0
			for t in xrange(1, combined_trends.shape[2]):
				if trends[e,f,t] != trends[e,f,t-1]:
					# Set End time for previous trend
					combined_trends_interval_end[e,f,combined_trend_idx] = t

					# Start next trend
					combined_trend_idx += 1
					combined_trends[e,f,combined_trend_idx] = trends[e,f,t]
					combined_trends_interval_start[e,f,combined_trend_idx] = t

	summed = np.sum(combined_trends == get_trend_idx('D'), 2)
	print summed
	print np.unravel_index(summed.argmax(), summed.shape)

	# print trends[5, 3, :]
	# print combined_trends[5, 3, :]
	# print combined_trends_interval_start[5, 3, :]
	# print combined_trends_interval_end[5, 3, :]
	# NUM_SIGN = 3 
	# NUM_EXAMPLES = 70
	# idx = NUM_SIGN*NUM_EXAMPLES
	# print trends[np.where(y==3)[0][0],3,:]
	# print combined_trends[np.where(y==3)[0][0],3,:]
	# print combined_trends_interval_start[np.where(y==3)[0][0],3,:]
	# print combined_trends_interval_end[np.where(y==3)[0][0],3,:]

	# support(combined_trends, None, 10)
	patterns = [None] * NUM_SIGNS
	pattern_counts = Counter()
	all_pattern_supports = [None] * NUM_SIGNS
	for i in xrange(NUM_SIGNS): 
		print 'Generating one patterns.....'
		one_patterns = generate_one_pattern(combined_trends)
		# support((combined_trends, combined_trends_interval_start, combined_trends_interval_end), 
		# 	'S:1-o;S:2-b;D:1-o;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1-b;D:1', 
		# 	w=5, minsup=20)
		# sys.exit()

		k = 2
		total_time = 0
		patterns[i] = [one_patterns]
		all_pattern_supports[i] = [None]
		while True and k <= 10:
			# generate_k_patterns
			print 'Generating patterns...'

			generation_start_time = time.time()
			total_time = 0
			new_patterns, total_time = generate_k_patterns(patterns[i][k-2], k, total_time)
			generation_end_time = time.time()
			print "k:", k, " -> New patterns:", len(new_patterns)

			print "Total time for generation: ", (generation_end_time-generation_start_time)
			print "Total time for comparison: ", (total_time)
			# prune_patterns
			print 'Pruning patterns...'
			combined_trends_sign = combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN,:,:]
			combined_trends_interval_start_sign = combined_trends_interval_start[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN,:,:]
			combined_trends_interval_end_sign = combined_trends_interval_end[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN,:,:]
			
			pruned_patterns, pattern_supports = prune_patterns((combined_trends_sign, combined_trends_interval_start_sign, combined_trends_interval_end_sign), 
											  new_patterns, minsup=20, w=5)

			all_pattern_supports[i].append(pattern_supports)
			for pattern in pattern_supports: 
				pattern_counts[pattern] += 1

			print "Pruned Patterns:", len(pruned_patterns)

			# increment k
			k += 1
			# If no k patterns, break
			if len(pruned_patterns) == 0:
				break
			print pruned_patterns.keys()[0]

			patterns[i].append(pruned_patterns)
			# patterns.append(new_patterns)

		print "pruned patterns: "
		print patterns[i] 
		print all_pattern_supports[i]

	ranked_patterns = chi_square(patterns, pattern_counts, all_pattern_supports)[:NUM_PATTERN_FEATURES]

	# cut some ranked_patterns out

	X_new = construct_feature_vectors(ranked_patterns, (combined_trends, combined_trends_interval_start, combined_trends_interval_end))

	# print ranked_patterns[:NUM_PATTERN_FEATURES]
	# print len(ranked_patterns)

	print X_new
	print y
	y = y[:NUM_SIGNS * EXAMPLES_PER_SIGN]

	svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
	svm_model.fit(X_new, y)
	y_predict_train = svm_model.predict(X_new)

	_, error = analysis.output_error(y_predict_train, y)
	print error




if __name__ == '__main__':
	seg_mining(use_all_signs=False)