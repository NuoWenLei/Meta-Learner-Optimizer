import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from structures.functional_layers import *
from helper_functions import *

class Learner():

	def __init__(self, args):

		super().__init__()
		self.filters = args["filters"]
		self.use_bias = args["use_bias"]
		self.img_x = args["img_x"]
		self.img_y = args["img_y"]
		self.padding = args["padding"]
		self.kernel_size = (args["kernel_size"], args["kernel_size"])
		self.num_blocks = args["num_blocks"]
		self.maxpool_size = (args["maxpool_size"], args["maxpool_size"])
		self.num_classes = args["num_classes"]

		self.model = self.get_learner_model()

		self.function_dict = {
			"conv2d": get_conv_func(self.padding, self.use_bias),
			"dense": get_dense_func(self.use_bias)
		}
		
	def reset_model(self):
		self.model = self.get_learner_model()

	# Create a fully function-based implementation of a learner model
	# based on its original structure
	def functional_learner(self, inputs, cI):

		# Initialize dictionary for storing weights by their Tensor name
		weights_dict = {}

		# trainable weights
		# I assign the weight from cI and the name from the trainable weights
		for n, a in zip(self.model.trainable_weights, cI):
			weights_dict[n.name] = a

		# non-trainable weights
		# This is one of the biggest problems with this functional implementation
		# Often times non-trainable weights are updated through a structural pattern,
		# which I'm not exactly sure how to store and implement yet.
		for w in self.model.non_trainable_weights:
			weights_dict[w.name] = w

		x = inputs

		# arrange weights by layer and kernel/bias
		for l in self.model.layers:

			# If there are no trainable weights,
			# just apply the layer on the input
			#
			# this is for layers like:
			# - ReLU
			# - Flatten
			# - MaxPool2D
			if len(l.trainable_weights) == 0:
				x = l(x)
				continue

			# Get all weights of a layer
			weight_list = [weights_dict[w.name] for w in l.weights]

			# Retrieve the corresponding functional layer from the dictionary.
			# Since the name of the layer is always in the name of the weights (Ex: "conv2d/kernel:0"),
			# I can just splice the layer name out.
			layer_func = self.function_dict[l.weights[0].name.split("/")[0].split("_")[0]] 

			# Apply layer function on inputs with the appropriate weights
			x = layer_func(x, weight_list)

		return x

	# Function for defining a Convolution block in learner
	def learner_block(self, inputs):
		x = Conv2D(self.filters, self.kernel_size, use_bias = self.use_bias, padding = self.padding)(inputs)
		x = ReLU()(x)
		x = MaxPool2D()(x)
		return x


	# Function for creating a learner model
	def get_learner_model(self):
		inputs = Input(shape = (self.img_x, self.img_y, 1))
		x = inputs
		for _ in range(self.num_blocks - 1):
			x = self.learner_block(x)
		x = Conv2D(self.filters, self.kernel_size, use_bias = self.use_bias, padding = self.padding)(x)
		x = ReLU()(x)
		flattened = Flatten()(x)
		output = Dense(self.num_classes, activation = "softmax", use_bias = self.use_bias)(flattened)
		return Model(inputs = inputs, outputs = output)