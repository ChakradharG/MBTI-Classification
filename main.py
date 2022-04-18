import numpy as np
from preprocessing import *
import argparse

np.random.seed = 1	# uncomment for better reproducibility



def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--l', help='length of histogram', type=int, default=1000)
    p.add_argument('--nc', help='Don\'t use the cached data', default=False, action='store_true')

    return p.parse_args()


def main(args):
	features, labels = get_data(args)	# pass --nc flag if running for the first time
	# labels = vectorize_labels(labels)	# one-hot vector encoding
	(x_trn, y_trn), (x_val, y_val), (x_tst, y_tst) = split_data(features, labels)

	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(max_iter=10000, C=100, class_weight='balanced')
	model.fit(x_trn, y_trn)
	yh = model.predict(x_trn)
	print(np.where(y_trn == yh)[0].shape[0]/y_trn.shape[0])
	yh = model.predict(x_val)
	print(np.where(y_val == yh)[0].shape[0]/y_val.shape[0])
	yh = model.predict(x_tst)
	print(np.where(y_tst == yh)[0].shape[0]/y_tst.shape[0])

	breakpoint()


if __name__ == '__main__':
	args = get_args()
	main(args)
