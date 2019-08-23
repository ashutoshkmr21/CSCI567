import numpy as np
import time
import random
from hmm import HMM
from collections import defaultdict


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	state_dict = defaultdict()
	word_dict = defaultdict()
	for ind,tag in enumerate(tags):
		state_dict[tag] = ind

	all_words = []
	state_state_count = defaultdict(dict)
	word_state_count = defaultdict(dict)

	start_tags = defaultdict()

	## Getting Transitions and Word State Counts
	for record in train_data:
		all_words += record.words
		start_tags[record.tags[0]]  = start_tags[record.tags[0]] + 1 if record.tags[0] in start_tags else 1
		for id in range(len(record.words)-1):
			t_tag  = record.tags[id]
			t_plus_1_tag = record.tags[id+1]
			word = record.words[id]

			try:
				state_state_count[t_tag][t_plus_1_tag] = state_state_count[t_tag][t_plus_1_tag] + 1
			except:
				state_state_count[t_tag][t_plus_1_tag] = 1
			try:
				word_state_count[word][t_tag] = word_state_count[word][t_tag] + 1
			except:
				word_state_count[word][t_tag] = 1
		try:
			word_state_count[record.words[-1]][record.tags[-1]] = word_state_count[record.words[-1]][record.tags[-1]] + 1
		except:
			word_state_count[record.words[-1]][record.tags[-1]] = 1

	S = len(state_state_count.keys())
	L = len(word_state_count.keys())

	## Initializing A and B
	A =  np.zeros((S,S))
	B = np.zeros((S,L))

	## Creating Word Dictionary
	for ind,word in enumerate(set(word_state_count.keys())):
		word_dict[word] = ind

	## Creating A and B
	for k1,v1 in state_state_count.items():
		# print(k1,' : ', v1)
		for k2,v2 in v1.items():
			A[state_dict[k1]][state_dict[k2]] = v2

	A = np.divide(A,np.sum(A,axis=1).reshape((S,1)))

	## Creating B
	for k1, v1 in word_state_count.items():
		# print(k1, ' : k1', v1)
		for k2, v2 in v1.items():
			B[state_dict[k2]][word_dict[k1]] = v2

	B = np.divide(B,np.sum(B,axis=0))

	pi = np.zeros((S))

	pi_sum = np.sum(list(start_tags.values()))


	## Creating pi
	for i in start_tags.keys():

		pi[state_dict[i]] = np.divide(start_tags[i],pi_sum)

	model = HMM( pi, A, B, word_dict, state_dict)


	###################################################
	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################

	obs_dict = model.obs_dict
	emission_mat = model.B
	biggest_key  = sorted(list(obs_dict.values()),reverse=True)[0]+1
	S = len(model.B[:,0])
	unk_col = np.ones((S,1))*(10**(-6))

	# Edit here
	for record in test_data:

		for word in record.words:
			if word not in model.obs_dict:
				emission_mat = np.append(emission_mat ,unk_col,axis=1)
				model.obs_dict[word] = biggest_key
				biggest_key+=1

	model.B = emission_mat

	for record in test_data:
		tagging.append(model.viterbi(record.words))


	###################################################
	return tagging

