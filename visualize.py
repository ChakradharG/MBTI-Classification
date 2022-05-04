import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def feature_images(features, labels):
	plt.figure()
	for i in range(20):
		# reshaping a 1D feature vector into 22*22 matrix to display as an image
		# since our feature vector is of length 500, height and width = int(sqrt(500)) = 22
		plt.imshow(np.resize(features[i], (22, 22)))
		plt.title(labels[i])
		plt.show()
		# plt.savefig(f'./data/feature_image_{i}.png')


def num_classes(labels):
	label_list = {}
	for label in labels:
		# counting the occurrences of each class in the dataset and storing it in a dictionary
		label_list[label] = label_list.get(label, 0) + 1
	plt.figure()
	plt.bar(label_list.keys(), label_list.values())
	plt.title('Number of examples of each class in the dataset')
	plt.show()
	# plt.savefig(f'./data/num_classes.png')


def display_conf_mat(model, model_name, x_tst, y_tst):
	y_pred = model.predict(x_tst)	# making predictions on test data
	conf_mat = confusion_matrix(y_tst, y_pred)
	ConfusionMatrixDisplay(
		confusion_matrix=conf_mat,
		display_labels=model.classes_
	).plot()
	plt.title(model_name)
	plt.show()
	# plt.savefig(f'./data/conf_mat_{model_name}.png')
