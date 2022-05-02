from preprocessing import *
from main import get_args, get_data
from matplotlib import pyplot as plt



def main(args):
	features, labels = get_data(args)

	# visualizing the first 20 examples as images
	plt.figure()
	for i in range(20):
		plt.imshow(np.resize(features[i], (22, 22)))
		plt.title(labels[i])
		plt.show()
		# plt.savefig(f'./data/{i}.png')

	# visualizing number of examples in each of the 16 classes
	label_list = {}
	for label in labels:
		label_list[label] = label_list.get(label, 0) + 1
	plt.figure()
	plt.bar(label_list.keys(), label_list.values())
	plt.show()
	# plt.savefig(f'./data/num_classes.png')
	

if __name__ == '__main__':
	args = get_args()
	main(args)
