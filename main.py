import numpy as np
from preprocessing import *
from hypertune import tune
from visualize import feature_images, num_classes, display_conf_mat
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

# np.random.seed = 1	# uncomment for better reproducibility



def get_args():
	# parse command line arguments. We can use these to run the models to our liking
	p = argparse.ArgumentParser()
	p.add_argument('--l', help='length of histogram', type=int, default=500)
	p.add_argument('--d', help='discard the top d words from dict', type=int, default=100)
	p.add_argument('--nc', help='don\'t use the cached data', default=False, action='store_true')
	p.add_argument('--ls', help='load saved model', default=False, action='store_true')
	p.add_argument('--lr', help='run logistic regression', default=False, action='store_true')
	p.add_argument('--lr_mxi', help='max_iter for lr', type=int, default=100)
	p.add_argument('--lr_c', help='parameter c for lr', type=int, default=500)
	p.add_argument('--svm', help='run SVM', default=False, action='store_true')
	p.add_argument('--svm_mxi', help='max_iter for svm', type=int, default=1000)
	p.add_argument('--svm_c', help='parameter c for svm', type=int, default=20)
	p.add_argument('--dtc', help='run decision tree', default=False, action='store_true')
	p.add_argument('--dtc_c', help='criterion for dtc', type=str, default='gini', choices=['gini', 'entropy'])
	p.add_argument('--dtc_mxd', help='max_depth for dtc', type=int, default=10)
	p.add_argument('--nb', help='run naive bayes', default=False, action='store_true')
	p.add_argument('--mlp', help='run MLP', default=False, action='store_true')
	p.add_argument('--mlp_mxi', help='max_iter for mlp', type=int, default=200)
	p.add_argument('--mlp_lri', help='inital learning rate for mlp', type=float, default=0.001)
	p.add_argument('--mlp_alp', help='regularization for mlp', type=float, default=0.01)
	p.add_argument('--all', help='run all models', default=False, action='store_true')
	p.add_argument('--hp', help='optimize hyper parameters', default=False, action='store_true')

	return p.parse_args()


def main(args):
	features, labels = get_data(args)	# pass --nc flag if running for the first time

	# feature_images(features, labels)	# visualizing the first 20 examples as images
	# num_classes(labels)	# visualizing number of examples in each of the 16 classes

	(x_trn, y_trn), (x_tst, y_tst) = split_data(features, labels)	# 70-30 split

	# depending upon the command line arguments passed, run the models
	if args.lr or args.all:
		if args.hp:	# whether to search for optimal hyperparameters
			args = tune(LogisticRegression(), 'lr', args, x_trn, y_trn)
		if args.ls:	# whether to load a saved model
			modelLR = load('./out/modelLR.joblib')
		else:	# whether to train a model from scratch
			modelLR = LogisticRegression(max_iter=args.lr_mxi, C=args.lr_c).fit(x_trn, y_trn)
		print('LR: ', modelLR.score(x_tst, y_tst))	# test accuracy
		display_conf_mat(modelLR, 'Logistic Regression', x_tst, y_tst)
		dump(modelLR, './out/modelLR.joblib')	#save to model

	if args.svm or args.all:
		if args.hp:	# whether to search for optimal hyperparameters
			args = tune(svm.SVC(), 'svm', args, x_trn, y_trn)
		if args.ls:	# whether to load a saved model
			modelSVM = load('./out/modelSVM.joblib')
		else:	# whether to train a model from scratch
			modelSVM = svm.SVC(max_iter=args.svm_mxi, C=args.svm_c).fit(x_trn,y_trn)
		print('SVM: ', modelSVM.score(x_tst, y_tst))	# test accuracy
		display_conf_mat(modelSVM, 'SVM', x_tst, y_tst)
		dump(modelSVM, './out/modelSVM.joblib')	#save to model

	if args.dtc or args.all:
		if args.hp:	# whether to search for optimal hyperparameters
			args = tune(DecisionTreeClassifier(), 'dtc', args, x_trn, y_trn)
		if args.ls:	# whether to load a saved model
			modelDTC = load('./out/modelDTC.joblib')
		else:	# whether to train a model from scratch
			modelDTC = DecisionTreeClassifier(criterion=args.dtc_c, max_depth=args.dtc_mxd).fit(x_trn,y_trn)
		print('DTC: ', modelDTC.score(x_tst, y_tst))	# test accuracy
		display_conf_mat(modelDTC, 'Decision Tree', x_tst, y_tst)
		dump(modelDTC, './out/modelDTC.joblib')	#save to model

	if args.nb or args.all:
		if args.hp:	# whether to search for optimal hyperparameters
			args = tune(GaussianNB(), 'nb', args, x_trn, y_trn)
		if args.ls:	# whether to load a saved model
			modelNB = load('./out/modelNB.joblib')
		else:	# whether to train a model from scratch
			modelNB = GaussianNB().fit(x_trn,y_trn)
		print('NB: ', modelNB.score(x_tst, y_tst))	# test accuracy
		display_conf_mat(modelNB, 'Naive Bayes', x_tst, y_tst)
		dump(modelNB, './out/modelNB.joblib')	#save to model

	if args.mlp or args.all:
		if args.hp:	# whether to search for optimal hyperparameters
			args = tune(MLPClassifier(), 'mlp', args, x_trn, y_trn)
		if args.ls:	# whether to load a saved model
			modelMLP = load('./out/modelMLP.joblib')
		else:	# whether to train a model from scratch
			modelMLP = MLPClassifier(
				learning_rate='adaptive',
				max_iter=args.mlp_mxi,
				learning_rate_init=args.mlp_lri,
				alpha=args.mlp_alp
			).fit(x_trn, y_trn)
		print('MLP: ', modelMLP.score(x_tst, y_tst))	# test accuracy
		display_conf_mat(modelMLP, 'Multilayer Perceptron (Neural Network)', x_tst, y_tst)
		dump(modelMLP, './out/modelMLP.joblib')	#save to model

if __name__ == '__main__':
	args = get_args()
	main(args)
