import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
import re
import sys

##decreasing: -0.3 < x <= 0
##very decreasing: -0.6 <= x <= -0.3 
##very very decreasing: -0.6 >= x
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
	patterns = set()

	##instead of iterating through all signals, go through only 3
	for e in xrange(D.shape[0]):
		for s in xrange(D.shape[1]):
			for t in xrange(D.shape[2]):
				if D[e,s,t] == -1:
					break
				patterns.add(get_trend(D[e,s,t]) + ':' + str(s+1))

	return patterns

def generate_k_patterns(patterns, k):
	new_patterns = set()
	if k == 2:
		for pattern_1 in patterns:
			for pattern_2 in patterns:
				if pattern_1 != pattern_2:
					signal_1 = pattern_1[2]
					signal_2 = pattern_2[2]

					if signal_1 != signal_2:
						# Generate 'overlap'
						new_patterns.add(pattern_1 + '-o;' + pattern_2)
						new_patterns.add(pattern_2 + '-o;' + pattern_1)

					new_patterns.add(pattern_1 + '-b;' + pattern_2)
					new_patterns.add(pattern_2 + '-b;' + pattern_1)
	else:
		for pattern_1 in patterns:
			for pattern_2 in patterns:
				if pattern_1 != pattern_2:
					sep_1_idx = pattern_1.rfind(';')
					sep_2_idx = pattern_2.rfind(';')
					prefix_1 = pattern_1[: sep_1_idx-2]
					prefix_2 = pattern_2[: sep_2_idx-2]

					if prefix_1 != prefix_2:
						continue

					suffix_1 = pattern_1[sep_1_idx+1:]
					suffix_2 = pattern_1[sep_2_idx+1:]

					signal_1 = prefix_1[-1]
					signal_2 = prefix_2[-1]
					relation_1 = pattern_1[sep_1_idx-1]
					relation_2 = pattern_2[sep_2_idx-1]

					if signal_1 != signal_2:
						# Generate 'overlap'
						new_patterns.add(pattern_1 + '-o;' + suffix_2)
						if not (relation_1 == 'o' and relation_2 == 'b'):
							new_patterns.add(pattern_2 + '-o;' + suffix_1)

					new_patterns.add(pattern_1 + '-b;' + suffix_2)
					if not (relation_1 == 'o' and relation_2 == 'b'):
						new_patterns.add(pattern_2 + '-b;' + suffix_1)

	return new_patterns

def pattern_exists_recurse(D, states, relations, state_idx, relation_idx, w, cur_window, prev_interval):
	##base case 
	if state_idx == len(states):
		return True

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
		recurse = True

		##update window and check if it's within window size
		if cur_window == None: 
			window_start = end
		else: 
			if end < window_start: 
				window_start = end

		window_end = start 	

		if window_end - window_start > w: 
			recurse = False 

		##check if pattern exists
		if prev_interval == None:
			recurse = True
		else:
			prev_start, prev_end = prev_interval

			# check b
			if relations[relation_idx] == 'b':
				if prev_end > start:
					recurse = False
			else:
				# check o
				if prev_end < start or start < prev_start:
					recurse = False
	
		##recursive case
		if recurse:
			if pattern_exists_recurse(D, states, relations, state_idx+1, relation_idx+1, w, (window_start, window_end),(start, end)):
				return True

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
		if current_sum >= minsup: 
			break

	return current_sum 	
	# return sum([pattern_exists((D[0][i,:,:], D[1][i,:,:], D[2][i,:,:]), P, w) for i in xrange(D[0].shape[0])])

def prune_patterns(D, patterns, minsup, w=20):
	pruned_patterns = set()

	for i, pattern in enumerate(patterns):
		if i % 10 == 0:
			print 'Done with', i
		if support(D, pattern, w, minsup) >= minsup:
			pruned_patterns.add(pattern)

	return pruned_patterns

def seg_mining(use_all_signs):
	# Constants
	STEADY_THRESHOLD = 0.1
	INCREASING_THRESHOLD = 0.3 
	VERY_INCREASING_THRESHOLD = 0.6 
	NUM_SIGNS = 1
	NUM_SIGNALS = 3
	EXAMPLES_PER_SIGN = 70

	# Loading data
	data = io.load_data(quality="low")
	X, y, class_names = preprocessing.create_data_tensor(data)	

	if not use_all_signs: 
		X = preprocessing.scale_spatially(X)[:NUM_SIGNS * EXAMPLES_PER_SIGN,:NUM_SIGNALS,:]
	else:  
		X = preprocessing.scale_spatially(X)

	# print X.shape 
	# sys.exit()
	# Computing fake slopes
	dX = np.roll(X, -1, 2) - X
	dX[:, :, -1] = dX[:, :, -2]

	# Computing trends
	binary_I_idx = (INCREASING_THRESHOLD >= dX) & (dX > STEADY_THRESHOLD)
	I_idx = np.where(binary_I_idx == 1)
	binary_VI_idx = (VERY_INCREASING_THRESHOLD >= dX) & (dX > INCREASING_THRESHOLD)
	VI_idx = np.where(binary_VI_idx == 1)
	WI_idx = np.where(dX > VERY_INCREASING_THRESHOLD)
	binary_D_idx = (-1*INCREASING_THRESHOLD <= dX) & (dX < -1*STEADY_THRESHOLD)
	D_idx = np.where(binary_D_idx == 1)
	binary_VD_idx = (-1*VERY_INCREASING_THRESHOLD <= dX) & (dX < -1*INCREASING_THRESHOLD)
	VD_idx = np.where(binary_VD_idx == 1)
	WD_idx = np.where(dX < -1*VERY_INCREASING_THRESHOLD)
	S_idx = np.where(np.abs(dX) <= STEADY_THRESHOLD)
	
	trends = np.zeros(dX.shape, dtype=np.int8)
	trends[I_idx] = get_trend_idx('I')
	trends[VI_idx] = get_trend_idx('VI')
	trends[WI_idx] = get_trend_idx('WI')
	trends[S_idx] = get_trend_idx('S')
	trends[D_idx] = get_trend_idx('D')
	trends[VD_idx] = get_trend_idx('VD')
	trends[WD_idx] = get_trend_idx('WD')


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
					combined_trends_interval_end[e,f,combined_trend_idx] = t-1

					# Start next trend
					combined_trend_idx += 1
					combined_trends[e,f,combined_trend_idx] = trends[e,f,t]
					combined_trends_interval_start[e,f,combined_trend_idx] = t

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
	print 'Generating one patterns.....'
	one_patterns = generate_one_pattern(combined_trends)

	k = 2
	patterns = [one_patterns]
	while True:
		# generate_k_patterns
		print 'Generating patterns...'
		new_patterns = generate_k_patterns(patterns[k-2], k)
		print k,len(new_patterns)

		# prune_patterns
		print 'Pruning patterns...'
		pruned_patterns = prune_patterns((combined_trends, combined_trends_interval_start, combined_trends_interval_end), 
										  new_patterns, minsup=40, w=5)

		print len(pruned_patterns)

		# increment k
		k += 1

		# If no k patterns, break
		if len(pruned_patterns) == 0:
			break

		patterns.append(pruned_patterns)
		# patterns.append(new_patterns)

	print "pruned patterns: "
	print patterns 
	# TODOS:
	# 1: Try numerical error for combining
if __name__ == '__main__':
	seg_mining(use_all_signs=False)