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
#2	20	2	400	95	8	0.01
STEADY_THRESHOLD = 0.1
WINDOW_SIZE = 2
MIN_SUPPORT = 20
K = 2
NUM_PATTERN_FEATURES = 400
NUM_SIGNS = 95
NUM_SIGNALS = 8
EXAMPLES_PER_SIGN = 49
PENALTY = 0.01

def process_sign(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end, patterns, pattern_counts, all_pattern_supports):
	print ''
	print 'For sign %d...' % i 
	print 'Generating one patterns.....'
	one_patterns = sq.generate_one_pattern(combined_trends)

	k = 2
	total_time = 0
	patterns[i] = [one_patterns]
	all_pattern_supports[i] = [None]
	while True and k <= K:
		# generate_k_patterns
		print 'Generating patterns...'

		generation_start_time = time.time()
		total_time = 0
		new_patterns, total_time = sq.generate_k_patterns(patterns[i][k-2], k, total_time)

		generation_end_time = time.time()
		print "k:", k, " -> New patterns:", len(new_patterns)

		print "Total time for generation: ", (generation_end_time-generation_start_time)
		print "Total time for comparison: ", (total_time)

		if len(new_patterns) == 0: break
		# prune_patterns
		print 'Pruning patterns...'
		# Compute position patterns
		combined_trends_sign = combined_trends[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:]
		combined_trends_interval_start_sign = combined_trends_interval_start[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:]
		combined_trends_interval_end_sign = combined_trends_interval_end[i*EXAMPLES_PER_SIGN:(i+1)*EXAMPLES_PER_SIGN, :,:]
		
		pruned_patterns, pattern_supports = sq.prune_patterns((combined_trends_sign, combined_trends_interval_start_sign, combined_trends_interval_end_sign), 
										  new_patterns, MIN_SUPPORT, WINDOW_SIZE)

		all_pattern_supports[i].append(pattern_supports)
		for pattern in pattern_supports:
			pattern_counts[pattern] += 1
		print "Pruned Patterns:", len(pruned_patterns)

		# increment k
		k += 1
		# If no k patterns, break
		if len(pruned_patterns) == 0:
			break
		#print pruned_patterns.keys()[0]

		patterns[i].append(pruned_patterns)
	return True

def seg_mining(use_all_signs):
	# Loading data
	data = io.load_data(quality="low")
	X, y, class_names = preprocessing.create_data_tensor(data)	
	X_train, y_train, X_test, y_test = preprocessing.create_train_test_split(X, y, test_size=0.3, shuffle=False)

	# for class_idx in np.unique(y):
	# 	print class_names[class_idx], np.where(class_idx == y)[0].shape[0]
	# import sys
	# sys.exit(1)

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
	dX_train = X_train 
	dX_test = X_test 

	# dX_train = np.roll(X_train, -1, 2) - X_train
	# dX_train[:, :, -1] = dX_train[:, :, -2]
	# dX_test = np.roll(X_test, -1, 2) - X_test
	# dX_test[:, :, -1] = dX_test[:, :, -2]

	combined_trends, combined_trends_interval_start, combined_trends_interval_end = sq.create_combined_trends_discritized(dX_train)
	combined_trends_test, combined_trends_interval_start_test, combined_trends_interval_end_test = sq.create_combined_trends_discritized(dX_test)

	patterns = [None] * NUM_SIGNS
	pattern_counts = Counter()
	all_pattern_supports = [None] * NUM_SIGNS

	threads = []
	for i in xrange(NUM_SIGNS):
		process_sign(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end,  patterns, pattern_counts, all_pattern_supports)
	# 	thread = Thread(target=process_sign, args=(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end,  patterns, pattern_counts, all_pattern_supports))
	# 	thread.start()
	# 	threads.append(thread)
	
	# for thread in threads:
	# 	thread.join()

		# process_sign(i, combined_trends, combined_trends_interval_start, combined_trends_interval_end, patterns, pattern_counts, all_pattern_supports)
		# for pattern_supports in all_pattern_supports[i]:
		# 	if pattern_supports is not None:
		# 		for pattern in pattern_supports:
		# 			pattern_counts[pattern] += 1

	# cut some ranked_patterns out
	print 'Classes',len(patterns), len(all_pattern_supports)
	print 'Iterations',len(patterns[1]), len(all_pattern_supports[1])
	# print 'patterns in iteration 1 ',len(patterns[1][0]), len(all_pattern_supports[1][0])
	
	ranked_patterns_all = sq.chi_square(patterns, pattern_counts, all_pattern_supports)

	# NUM_PATTERN_FEATURES_ARRAY = [75, 100, 200, 300, 400, 500, 750, 1000, 1500]
	# NUM_PATTERN_FEATURES_ARRAY = [400, 500, 750, 1000, 1500, 2000, 2500]
	NUM_PATTERN_FEATURES_ARRAY = [NUM_PATTERN_FEATURES]
	for num_pattern_features in NUM_PATTERN_FEATURES_ARRAY:
		print '---------------------=================---------------------'
		ranked_patterns = ranked_patterns_all[:num_pattern_features]
		print len(ranked_patterns_all),'->',len(ranked_patterns)
		
		X_train_new = sq.construct_feature_vectors(ranked_patterns, \
			(combined_trends, combined_trends_interval_start, combined_trends_interval_end), EXAMPLES_PER_SIGN)
		X_test_new = sq.construct_feature_vectors(ranked_patterns, \
			(combined_trends_test, combined_trends_interval_start_test, combined_trends_interval_end_test), 70 - EXAMPLES_PER_SIGN)

		svm_model = svm.SVC(C=PENALTY, kernel="linear", decision_function_shape='ovr')
		svm_model.fit(X_train_new, y_train)
		y_predict_train = svm_model.predict(X_train_new)
		y_predict = svm_model.predict(X_test_new)

		# np.save('../data/seq_mining_features/X_train-%d.npy'%num_pattern_features, X_train_new)
		# np.save('../data/seq_mining_features/X_test-%d.npy'%num_pattern_features, X_test_new)
		# np.save('../data/seq_mining_features/y_train-%d.npy'%num_pattern_features, y_train)
		# np.save('../data/seq_mining_features/y_test-%d.npy'%num_pattern_features, y_test)

		analysis.run_analyses(y_predict_train, y_train, y_predict, y_test, class_names)
		print 'STEADY_THRESHOLD',STEADY_THRESHOLD
		print 'WINDOW_SIZE',WINDOW_SIZE
		print 'MIN_SUPPORT',MIN_SUPPORT
		print 'K',K
		print 'NUM_PATTERN_FEATURES',num_pattern_features
		print 'NUM_SIGNS',NUM_SIGNS
		print 'NUM_SIGNALS',NUM_SIGNALS
		print 'EXAMPLES_PER_SIGN',EXAMPLES_PER_SIGN
		print 'PENALTY',PENALTY
	print '---------------------=================---------------------'


if __name__ == '__main__':
	seg_mining(use_all_signs=False)