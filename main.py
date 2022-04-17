import numpy as np
import pandas as pd
from words import Words



def get_data(cached=False):
	if cached:
		df = np.load('./data/cleaned.npz', allow_pickle=True)
		posts = df['posts']
		labels = df['labels']
	else:
		df = pd.read_csv('./data/mbti_1.csv')
		posts = np.array([*map(lambda s: Words(s), df['posts'])])
		labels = np.array(df['type'])
		np.savez('./data/cleaned', posts=posts, labels=labels)

	return posts, labels


def split_data(posts, labels):
	N = posts.shape[0]

	idx = np.arange(N)
	trn = np.random.choice(np.arange(idx.shape[0]), int(N*0.6), replace=False)

	idx = np.delete(idx, trn)
	val_i = np.random.choice(np.arange(idx.shape[0]), int(N*0.2), replace=False)
	val = idx[val_i]

	idx = np.delete(idx, val_i)
	tst = idx

	return (posts[trn], labels[trn]), (posts[val], labels[val]), (posts[tst], labels[tst])


def main():
	posts, labels = get_data(True)	# get_data() if running for the first time
	(x_trn, y_trn), (x_val, y_val), (x_tst, y_tst) = split_data(posts, labels)

	breakpoint()


if __name__ == '__main__':
	main()
