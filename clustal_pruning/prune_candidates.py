import numpy as np
import subprocess
import time
from joblib import Parallel, delayed  
import multiprocessing
import cPickle as pickle
import sys
import os
from os import listdir
from os.path import isfile, join

test_file='sampled_seqs.txt'
with open('../residue_lm/data/residue_lstm_input.txt','r') as f:
	seed_seqs=f.read().split('\n')[:-1]

with open(test_file,'r') as f:
	seqs=list(set(f.read().split('\n')[:-1]))
	test_seqs=[t for t in seqs if len(t)>8 and len(t)<21] #pretty harsh thresholds

num_seeds, num_tests=len(seed_seqs), len(test_seqs)
print num_seeds, num_tests

def get_similar_seqs(seq_no):

	test_seq=test_seqs[seq_no]
	dist1, dist2=np.zeros(num_seeds), np.zeros(num_seeds)
	for i in range(num_seeds):
		clustal_cmd="bash clustal.sh %s %s %d"%(seed_seqs[i], test_seq, seq_no)
		process = subprocess.Popen(clustal_cmd.split(), stdout=subprocess.PIPE)
		output = map(float, process.communicate()[0].split('\n')[:-1])
		dist1[i], dist2[i]= output[0], output[1]
		
	thresh_set1, thresh_set2=set(np.where(dist1>0.6)[0]), set(np.where(dist2>0.4)[0])
	similar=thresh_set1|thresh_set2
	if len(similar)>0:
		print '#similar:', len(similar)
		print 'seq:', test_seq
		for each in similar:
			print seed_seqs[each], dist1[each], dist2[each]
		print '------------------------------------------------'
	return similar

num_cores = multiprocessing.cpu_count()
similar_set=Parallel(n_jobs=num_cores, verbose=15)(delayed(get_similar_seqs)(i) for i in range(len(test_seqs)))

similar_set=np.array([len(s) for s in similar_set])
print len(similar_set)

with open('distinct_seqs.txt','w') as f:
	for idx in list(np.where(similar_set==0)[0]):
		f.write(test_seqs[idx]+'\n')





