import numpy as np
import util.analysis as analysis
import util.io as io
import util.preprocessing as preprocessing
import seq_mining as sq
import re
import sys
import time
from sklearn import svm

from collections import Counter

from threading import Thread

STEADY_THRESHOLD = 0.1
WINDOW_SIZE = 5
MIN_SUPPORT = 20
K = 5
NUM_PATTERN_FEATURES = 500
NUM_SIGNS = 5
NUM_SIGNALS = 8
EXAMPLES_PER_SIGN = 49
PENALTY = 0.1


def candidate_generation_iteration(D, patterns, supports, pattern_counts, RANGE, i, k):
	print 'Generating patterns...'
	combined_trends, combined_trends_interval_start, combined_trends_interval_end = D

	generation_start_time = time.time()
	total_time = 0
	new_patterns, total_time = sq.generate_k_patterns(patterns[k-2], k, total_time)

	generation_end_time = time.time()
	print "k:", k, " -> New patterns:", len(new_patterns)

	print "Total time for generation: ", (generation_end_time-generation_start_time)
	print "Total time for comparison: ", (total_time)

	if len(new_patterns) == 0: 
		return False
	# prune_patterns
	print 'Pruning patterns...'

	#
	# combined_trends_sign = combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:]
	# combined_trends_interval_start_sign = combined_trends_interval_start[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:]
	# combined_trends_interval_end_sign = combined_trends_interval_end[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:]

	# Compute position patterns
	combined_trends_sign = combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, RANGE,:]
	combined_trends_interval_start_sign = combined_trends_interval_start[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, RANGE,:]
	combined_trends_interval_end_sign = combined_trends_interval_end[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, RANGE,:]
	
	pruned_patterns, pattern_supports = sq.prune_patterns((combined_trends_sign, combined_trends_interval_start_sign, combined_trends_interval_end_sign), 
									  new_patterns, MIN_SUPPORT, WINDOW_SIZE)
	
	
	for pattern in pattern_supports:
		pattern_counts[pattern] += 1
	print "Pruned Patterns:", len(pruned_patterns)

	# If no k patterns, break
	if len(pruned_patterns) == 0:
		return False

	supports.append(pattern_supports)
	patterns.append(pruned_patterns)
	return True

def process_sign(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end, patterns, pattern_counts, all_pattern_supports):

	POSITION_RANGE = [0,1,2]
	ROTATION_RANGE = [3]
	FINGER_RANGE = [4,5,6,7]	
	print ''
	print 'For sign %d...' % i 
	print 'Generating one patterns.....'

	k = 2
	total_time = 0

	position_patterns = [sq.generate_one_pattern(combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, POSITION_RANGE,:])]
	rotation_patterns = [sq.generate_one_pattern(combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, ROTATION_RANGE,:])]
	finger_patterns = [sq.generate_one_pattern(combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, FINGER_RANGE,:])]

	position_supports = [None]
	rotation_supports = [None]
	finger_supports = [None]

	# patterns[i] = [sq.generate_one_pattern(combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:])]
	# all_pattern_supports[i] = [None]

	position_patterns_generated = True
	rotation_patterns_generated = True
	finger_patterns_generated = True
	while k <= K:
		# generate_k_patterns
		if position_patterns_generated:
			position_patterns_generated = candidate_generation_iteration((combined_trends, combined_trends_interval_start, combined_trends_interval_end),
										 position_patterns, position_supports, pattern_counts, POSITION_RANGE, i, k)
		if rotation_patterns_generated:
			rotation_patterns_generated = candidate_generation_iteration((combined_trends, combined_trends_interval_start, combined_trends_interval_end),
										 rotation_patterns, rotation_supports, pattern_counts, ROTATION_RANGE, i, k)
		if finger_patterns_generated:
			finger_patterns_generated = candidate_generation_iteration((combined_trends, combined_trends_interval_start, combined_trends_interval_end),
										 finger_patterns, finger_supports, pattern_counts, FINGER_RANGE, i, k)
		# increment k
		k += 1
		if not (position_patterns_generated or rotation_patterns_generated or finger_patterns_generated):
			break

	patterns[i] = position_patterns + rotation_patterns + finger_patterns
	all_pattern_supports[i] = position_supports + rotation_supports + finger_supports
	return True

def seg_mining(use_all_signs):
	# Loading data
	data = io.load_data(quality="low")
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=False)

	if not use_all_signs: 
		# TODO: change not break
		X_train = preprocessing.scale_spatially(X_train)[:NUM_SIGNS * EXAMPLES_PER_SIGN,:NUM_SIGNALS,:]
		y_train = y_train[:NUM_SIGNS * EXAMPLES_PER_SIGN]
		X_test = preprocessing.scale_spatially(X_test)[:NUM_SIGNS * (70 - EXAMPLES_PER_SIGN),:NUM_SIGNALS,:]
		y_test = y_test[:NUM_SIGNS * (70 - EXAMPLES_PER_SIGN)]
	else:  
		X_train = preprocessing.scale_spatially(X_train)
		X_test = preprocessing.scale_spatially(X_test)

	# Computing fake slopes
	# dX_train = X_train 
	# dX_test = X_test 

	dX_train = np.roll(X_train, -1, 2) - X_train
	dX_train[:, :, -1] = dX_train[:, :, -2]
	dX_test = np.roll(X_test, -1, 2) - X_test
	dX_test[:, :, -1] = dX_test[:, :, -2]

	combined_trends, combined_trends_interval_start, combined_trends_interval_end = sq.create_combined_trends(dX_train)
	combined_trends_test, combined_trends_interval_start_test, combined_trends_interval_end_test = sq.create_combined_trends(dX_test)

	patterns = [None] * NUM_SIGNS
	pattern_counts = Counter()
	all_pattern_supports = [None] * NUM_SIGNS

	threads = []
	for i in xrange(NUM_SIGNS):
		thread = Thread(target=process_sign, args=(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end,  patterns, pattern_counts, all_pattern_supports))
		thread.start()
		threads.append(thread)
	
	for thread in threads:
		thread.join()

		# process_sign(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end, patterns, pattern_counts, all_pattern_supports)
		# for pattern_supports in all_pattern_supports[i]:
		# 	if pattern_supports is not None:
		# 		for pattern in pattern_supports:
		# 			pattern_counts[pattern] += 1

	# cut some ranked_patterns out
	print 'Classes',len(patterns), len(all_pattern_supports)
	print 'Iterations',len(patterns[1]), len(all_pattern_supports[1])
	# print 'patterns in iteration 1 ',len(patterns[1][0]), len(all_pattern_supports[1][0])
	
	ranked_patterns = sq.chi_square(patterns, pattern_counts, all_pattern_supports)
	print len(ranked_patterns),
	ranked_patterns = ranked_patterns[:NUM_PATTERN_FEATURES]
	print '->',len(ranked_patterns)

	
	X_train_new = sq.construct_feature_vectors(ranked_patterns, \
		(combined_trends, combined_trends_interval_start, combined_trends_interval_end), EXAMPLES_PER_SIGN)
	X_test_new = sq.construct_feature_vectors(ranked_patterns, \
		(combined_trends_test, combined_trends_interval_start_test, combined_trends_interval_end_test), 70 - EXAMPLES_PER_SIGN)

	svm_model = svm.SVC(C=PENALTY, kernel="linear", decision_function_shape='ovr')
	svm_model.fit(X_train_new, y_train)
	y_predict_train = svm_model.predict(X_train_new)
	y_predict = svm_model.predict(X_test_new)

	analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names)
	print 'STEADY_THRESHOLD',STEADY_THRESHOLD
	print 'WINDOW_SIZE',WINDOW_SIZE
	print 'MIN_SUPPORT',MIN_SUPPORT
	print 'K',K
	print 'NUM_PATTERN_FEATURES',NUM_PATTERN_FEATURES
	print 'NUM_SIGNS',NUM_SIGNS
	print 'NUM_SIGNALS',NUM_SIGNALS
	print 'EXAMPLES_PER_SIGN',EXAMPLES_PER_SIGN
	print 'PENALTY',PENALTY


if __name__ == '__main__':
	seg_mining(use_all_signs=False)