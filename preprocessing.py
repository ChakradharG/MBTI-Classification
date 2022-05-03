import numpy as np
import pandas as pd
from words import Words
from tqdm import trange



def get_data(args):
	if args.nc:
		df = pd.read_csv('./data/mbti_1.csv')
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
		)
	else:
		df = np.load('./data/cleaned.npz', allow_pickle=True)
		features = df['features']
		labels = df['labels']
		args.l = df['l']

	return features, labels


def split_data(features, labels, val_split=False):
	N = features.shape[0]

	if val_split:
		idx = np.arange(N)
		trn = np.random.choice(np.arange(idx.shape[0]), int(N*0.6), replace=False)
		idx = np.delete(idx, trn)
		val_i = np.random.choice(np.arange(idx.shape[0]), int(N*0.2), replace=False)
		val = idx[val_i]
		idx = np.delete(idx, val_i)
		tst = idx
		return (features[trn], labels[trn]), (features[val], labels[val]), (features[tst], labels[tst])
	else:
		idx = np.arange(N)
		trn = np.random.choice(np.arange(idx.shape[0]), int(N*0.7), replace=False)
		idx = np.delete(idx, trn)
		tst = idx
		return (features[trn], labels[trn]), (features[tst], labels[tst])


def vectorize_labels(labels):
	lst = [
		'ESFP', 'ESFJ', 'ESTP', 'ESTJ',
		'ENFP', 'ENFJ', 'ENTP', 'ENTJ',
		'ISFP', 'ISFJ', 'ISTP', 'ISTJ',
		'INFP', 'INFJ', 'INTP', 'INTJ'
	]

	labels_vect = np.zeros((labels.shape[0], len(lst)), dtype=np.uint8)
	for i, label in enumerate(labels):
		labels_vect[i, lst.index(label)] = 1
	
	return labels_vect


def compute_BoW(args, posts):
	temp = {}
	for post in posts:
		for word in post.words:
			temp[word] = temp.get(word, 0) + 1

	sorted_keys = sorted(temp, key=temp.get, reverse=True)
	BoW = {}
	N = temp[sorted_keys[0]]

	for i in range(args.d, args.l):
		BoW[sorted_keys[i]] = (i-args.d, temp[sorted_keys[i]]/N)

	return BoW


def extract_features(args, posts, dictionary):
	features = np.zeros((posts.shape[0], args.l))

	for i in trange(len(posts), desc='Feature Extraction'):
		for word, (index, doc_freq) in dictionary.items():
			features[i, index] = posts[i].words.count(word)/doc_freq
		features[i] = features[i]/(features[i].sum() + 1)
	
	return features
