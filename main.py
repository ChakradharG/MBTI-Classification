import numpy as np
from preprocessing import *
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

# np.random.seed = 1	# uncomment for better reproducibility



def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--l', help='length of histogram', type=int, default=500)
    p.add_argument('--d', help='discard the top d words from dict', type=int, default=100)
    p.add_argument('--nc', help='don\'t use the cached data', default=False, action='store_true')
    p.add_argument('--ls', help='load saved model', default=False, action='store_true')
    p.add_argument('--lr', help='run logistic regression', default=False, action='store_true')
    p.add_argument('--svm', help='run SVM', default=False, action='store_true')
    p.add_argument('--dtc', help='run decision tree', default=False, action='store_true')
    p.add_argument('--nb', help='run naive bayes', default=False, action='store_true')
    p.add_argument('--mlp', help='run MLP', default=False, action='store_true')
    p.add_argument('--all', help='run all models', default=False, action='store_true')

    return p.parse_args()


def main(args):
	features, labels = get_data(args)	# pass --nc flag if running for the first time
	(x_trn, y_trn), (x_tst, y_tst) = split_data(features, labels)

	if args.lr or args.all:
		if args.ls:
			modelLR = load('./out/modelLR.joblib')
		else:
			modelLR = LogisticRegression(max_iter=10000, C=550).fit(x_trn, y_trn)
		print('LR: ', modelLR.score(x_tst, y_tst))	# test accuracy
		dump(modelLR, './out/modelLR.joblib')

	if args.svm or args.all:
		if args.ls:
			modelSVM = load('./out/modelSVM.joblib')
		else:
			modelSVM = svm.SVC().fit(x_trn,y_trn)
		print('SVM: ', modelSVM.score(x_tst, y_tst))	# test accuracy
		dump(modelSVM, './out/modelSVM.joblib')

	if args.dtc or args.all:
		if args.ls:
			modelDTC = load('./out/modelDTC.joblib')
		else:
			modelDTC = DecisionTreeClassifier().fit(x_trn,y_trn)
		print('DTC: ', modelDTC.score(x_tst, y_tst))	# test accuracy
		dump(modelDTC, './out/modelDTC.joblib')

	if args.nb or args.all:
		if args.ls:
			modelNB = load('./out/modelNB.joblib')
		else:
			modelNB = GaussianNB().fit(x_trn,y_trn)
		print('NB: ', modelNB.score(x_tst, y_tst))	# test accuracy
		dump(modelNB, './out/modelNB.joblib')

	if args.mlp or args.all:
		if args.ls:
			modelMLP = load('./out/modelMLP.joblib')
		else:
			modelMLP = MLPClassifier(max_iter=1000).fit(x_trn, y_trn)
		print('MLP: ', modelMLP.score(x_tst, y_tst))	# test accuracy
		dump(modelMLP, './out/modelMLP.joblib')

if __name__ == '__main__':
	args = get_args()
	main(args)
