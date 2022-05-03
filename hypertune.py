from sklearn.model_selection import GridSearchCV



def tune(model, model_name, args, x_trn, y_trn):
	if model_name == 'lr':
		params = {
			'max_iter': [100, 200, 500],
			'C': [20, 200, 250, 500]
		}
	elif model_name == 'svm':
		params = {
			'max_iter': [500, 1000, 2000],
			'C': [20, 250, 1000]
		}
	elif model_name == 'dtc':
		params = {
			'criterion': ['gini', 'entropy'],
			'max_depth': [3, 5, 10, 20, 50]
		}
	elif model_name == 'mlp':
		params = {
			'max_iter': [100, 200, 500],
			'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
			'alpha': [0.001, 0.01, 0.1]
		}

	clf = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=4).fit(x_trn, y_trn)
	print(clf.best_params_)

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
