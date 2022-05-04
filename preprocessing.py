import numpy as np
import pandas as pd
from words import Words
from tqdm import trange	# for progress bar



def get_data(args):
	# load the raw data and pre-process it
	if args.nc:	# whether to redo the feature extraction
		df = pd.read_csv('./data/mbti_1.csv')
		# converting a single string containing the person's posts into a list of words (tokens)
		posts = np.array([*map(lambda s: Words(s), df['posts'])])
		# print(posts[0].words)	# visualizing cleaned data
		labels = np.array(df['type'])
		dictionary = compute_BoW(args, posts)
		features = extract_features(args, posts, dictionary)
		np.savez(
			'./data/cleaned',
			posts=posts,
			labels=labels,
			dictionary=dictionary,
			features=features,
			l=args.l
		)	# save the extracted features etc for future use
	else:	# load already saved features, dont recompute
		df = np.load('./data/cleaned.npz', allow_pickle=True)
		features = df['features']
		labels = df['labels']
		args.l = df['l']

	return features, labels


def split_data(features, labels, val_split=False):
	# train-validation-test split
	N = features.shape[0]

	if val_split:	# if validation set required, then do 60-20-20 split
		idx = np.arange(N)	# all the available indices
		# select 60% of the indices randomly for train set
		trn = np.random.choice(np.arange(idx.shape[0]), int(N*0.6), replace=False)
		# delete already selected indices for train set
		idx = np.delete(idx, trn)
		# select 20% of the indices randomly for validation set
		val_i = np.random.choice(np.arange(idx.shape[0]), int(N*0.2), replace=False)
		# store the actual indices
		val = idx[val_i]
		# delete already selected indices for val set
		idx = np.delete(idx, val_i)
		# store the remaining indices into test set
		tst = idx
		return (features[trn], labels[trn]), (features[val], labels[val]), (features[tst], labels[tst])
	else:	# if validation set not required, then do 70-30 split
		idx = np.arange(N)	# all the available indices
		# select 70% of the indices randomly for train set
		trn = np.random.choice(np.arange(idx.shape[0]), int(N*0.7), replace=False)
		# delete already selected indices for train set
		idx = np.delete(idx, trn)
		# store the remaining indices into test set
		tst = idx
		return (features[trn], labels[trn]), (features[tst], labels[tst])


def vectorize_labels(labels):
	# convert labels into one-hot encoded vectors
	lst = [
		'ESFP', 'ESFJ', 'ESTP', 'ESTJ',
		'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
		'ISFP', 'ISFJ', 'ISTP', 'ISTJ',
		'INFP', 'INFJ', 'INTP', 'INTJ'
	]

	# creating empty np.array of shape (num_examples, num_classes(16))
	labels_vect = np.zeros((labels.shape[0], len(lst)), dtype=np.uint8)
	for i, label in enumerate(labels):
		labels_vect[i, lst.index(label)] = 1
	
	return labels_vect


def compute_BoW(args, posts):
	# compute the bag-of-words dictionary/vocabulary
	temp = {}
	for post in posts:
		for word in post.words:
			# counting the occurrence of each individual word in the dataset
			temp[word] = temp.get(word, 0) + 1

	# sorting the dictionary based on the decreasing order of number of occurrences of words (keys)
	sorted_keys = sorted(temp, key=temp.get, reverse=True)
	BoW = {}
	N = temp[sorted_keys[0]]	# max number of occurrences, for normaliation

	# args.d : remove the d most occurring words (stop words)
	# args.l : length of the feature vector
	for i in range(args.d, args.l):
		# storing the index of the word for the feature vector and its term-frequency
		BoW[sorted_keys[i]] = (i-args.d, temp[sorted_keys[i]]/N)

	return BoW


def extract_features(args, posts, dictionary):
	# converting the lists of words in posts into vectors of length args.l
	features = np.zeros((posts.shape[0], args.l))

	# looping over every example
	for i in trange(len(posts), desc='Feature Extraction'):
		for word, (index, doc_freq) in dictionary.items():
			# storing the number of times the particular word has occurred at the appropriate index
			features[i, index] = posts[i].words.count(word)/doc_freq
		features[i] = features[i]/(features[i].sum() + 1)	# normalization
	
	return features
