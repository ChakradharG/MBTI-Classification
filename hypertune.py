from sklearn.model_selection import GridSearchCV



def tune(model, model_name, args, x_trn, y_trn):
	# depending upon the model, choose the hyperparameters to search over
	if model_name == 'lr':
		params = {
			'max_iter': [100, 200, 500],	# max number of iterations
			'C': [20, 200, 250, 500]	# regularization penalty C (inverse of lambda)
		}
	elif model_name == 'svm':
		params = {
			'max_iter': [500, 1000, 2000],	# max number of iterations
			'C': [20, 250, 1000]	# regularization penalty C (inverse of lambda)
		}
	elif model_name == 'dtc':
		params = {
			'criterion': ['gini', 'entropy'],	# impurity criterion
			'max_depth': [3, 5, 10, 20, 50]		# max depth the tree can grow to
		}
	elif model_name == 'mlp':
		params = {
			'max_iter': [100, 200, 500],	# max number of iterations
			'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],	# initial learning rate
			'alpha': [0.001, 0.01, 0.1]		# regularization factor alpha
		}

	clf = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=4).fit(x_trn, y_trn)
	print(clf.best_params_)

	# save the best hyperparameter values into the args object and return it
	if model_name == 'lr':
		args.lr_mxi = clf.best_params_['max_iter']
		args.lr_c = clf.best_params_['C']
	elif model_name == 'svm':
		args.svm_mxi = clf.best_params_['max_iter']
		args.svm_c = clf.best_params_['C']
	elif model_name == 'dtc':
		args.dtc_c = clf.best_params_['criterion']
		args.dtc_mxd = clf.best_params_['max_depth']
	elif model_name == 'mlp':
		args.mlp_mxi = clf.best_params_['max_iter']
		args.mlp_lri = clf.best_params_['learning_rate_init']
		args.mlp_alp = clf.best_params_['alpha']
	
	return args
