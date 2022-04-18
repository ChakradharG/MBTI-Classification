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
	(x_trn, y_trn), (x_val, y_val), (x_tst, y_tst) = split_data(features, labels)

	breakpoint()


if __name__ == '__main__':
	args = get_args()
	main(args)
