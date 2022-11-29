import numpy as np
from preprocessing import *
from visualize import feature_images, num_classes, display_conf_mat
import argparse
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump, load
from tqdm import tqdm

# np.random.seed = 1	# uncomment for better reproducibility



def get_args():
	# parse command line arguments. We can use these to run the models to our liking
	p = argparse.ArgumentParser()
	p.add_argument('--l', help='length of histogram', type=int, default=500)
	p.add_argument('--d', help='discard the top d words from dict', type=int, default=100)
	p.add_argument('--nc', help='don\'t use the cached data', default=False, action='store_true')
	p.add_argument('--ls', help='load saved model', default=False, action='store_true')
	p.add_argument('--t', help='number of weak classifiers for adaboost', type=int, default=500)

	return p.parse_args()


def main(args):
	features, labels = get_data(args)	# pass --nc flag if running for the first time

	# feature_images(features, labels)	# visualizing the first 20 examples as images
	# num_classes(labels)	# visualizing number of examples in each of the 16 classes

	(x_trn, y_trn), (x_tst, y_tst) = split_data(features, labels)	# 70-30 split
	ei = np.array([int(i[0] == 'I') for i in labels])
	ns = np.array([int(i[1] == 'S') for i in labels])
	ft = np.array([int(i[2] == 'T') for i in labels])
	jp = np.array([int(i[3] == 'P') for i in labels])
	N = labels.shape[0]
	Wei = np.array([1 - ei.sum()/N, ei.sum()/N])
	Wns = np.array([1 - ns.sum()/N, ns.sum()/N])
	Wft = np.array([1 - ft.sum()/N, ft.sum()/N])
	Wjp = np.array([1 - jp.sum()/N, jp.sum()/N])

	y_tst_ei = np.array([int(i[0] == 'I') for i in y_tst])
	y_tst_ns = np.array([int(i[1] == 'S') for i in y_tst])
	y_tst_ft = np.array([int(i[2] == 'T') for i in y_tst])
	y_tst_jp = np.array([int(i[3] == 'P') for i in y_tst])

	if args.ls:	# whether to load a saved model
		model_ei = load('./model_ei.joblib')
		model_ns = load('./model_ns.joblib')
		model_ft = load('./model_ft.joblib')
		model_jp = load('./model_jp.joblib')
	else:	# whether to train a model from scratch
		y_trn_ei = np.array([int(i[0] == 'I') for i in y_trn])
		y_trn_ns = np.array([int(i[1] == 'S') for i in y_trn])
		y_trn_ft = np.array([int(i[2] == 'T') for i in y_trn])
		y_trn_jp = np.array([int(i[3] == 'P') for i in y_trn])
		model_ei = AdaBoostClassifier(n_estimators=args.t).fit(x_trn, y_trn_ei, Wei[y_trn_ei])
		print('model_ei: ', model_ei.score(x_tst, y_tst_ei))
		model_ns = AdaBoostClassifier(n_estimators=args.t).fit(x_trn, y_trn_ns, Wns[y_trn_ns])
		print('model_ns: ', model_ns.score(x_tst, y_tst_ns))
		model_ft = AdaBoostClassifier(n_estimators=args.t).fit(x_trn, y_trn_ft, Wft[y_trn_ft])
		print('model_ft: ', model_ft.score(x_tst, y_tst_ft))
		model_jp = AdaBoostClassifier(n_estimators=args.t).fit(x_trn, y_trn_jp, Wjp[y_trn_jp])
		print('model_jp: ', model_jp.score(x_tst, y_tst_jp))
		dump(model_ei, './model_ei.joblib')
		dump(model_ns, './model_ns.joblib')
		dump(model_ft, './model_ft.joblib')
		dump(model_jp, './model_jp.joblib')

	correct = 0
	total = x_tst.shape[0]

	batch_bar = tqdm(total=total, dynamic_ncols=True, leave=False, position=0, desc="Test")

	for i in range(total):
		pred = ''
		pred += ['E', 'I'][model_ei.predict(x_tst[i].reshape(1, -1))[0]]
		pred += ['N', 'S'][model_ns.predict(x_tst[i].reshape(1, -1))[0]]
		pred += ['F', 'T'][model_ft.predict(x_tst[i].reshape(1, -1))[0]]
		pred += ['J', 'P'][model_jp.predict(x_tst[i].reshape(1, -1))[0]]

		if pred == y_tst[i]:
			correct += 1

		batch_bar.set_postfix(acc=f"{correct / (i + 1):.4f}")
		batch_bar.update()

	batch_bar.close()
	print(correct/total)
	# display_conf_mat(modelADA, 'AdaBoost', x_tst, y_tst)


if __name__ == '__main__':
	args = get_args()
	main(args)
