import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras.utils import conv_utils, tf_utils

def get_conv_func(padding, use_bias):

	# Functional implementation of the layer Conv2D to avoid weight assignment

	def functional_conv2d(inputs, weights, padding = padding, use_bias = use_bias):

		# Load kernel and bias based on whether USE_BIAS is true
		kernel = weights[0]
		if use_bias:
			bias = weights[1]

		# Rank refers to how many dimensions the convolution involves
		# Since we're doing Conv2D, rank = 2
		rank = 2

		# We don't want the convolution to skip data, so strides = 1
		strides = conv_utils.normalize_tuple(1, rank, 'strides')

		# Just use default for dilation rate.
		# It has something to do with the magnitude of weights as the filter number rises
		dilation_rate = conv_utils.normalize_tuple(1, rank, 'dilation_rate')

		# data format refers to the shape of the input
		# default is NHWC and is what I'm using
		# - Number of samples
		# - Height of image
		# - Width of image
		# - Channels
		data_format = conv_utils.convert_data_format(conv_utils.normalize_data_format(None), rank + 2)

		# Call convolution function
		output = tf.nn.convolution(
			tf.cast(inputs, tf.float32),
			kernel,
			strides = list(strides),
			padding = padding,
			dilations = list(dilation_rate),
			data_format = data_format,
			name = "conv2d"
		)

		# Add bias if use_bias is True
		if use_bias:
			output = tf.nn.bias_add(
				output,
				bias,
				data_format = data_format
			)

		return output

	return functional_conv2d

def get_dense_func(use_bias):

	# Functional implementation of the Dense layer to avoid weight assignment

	def functional_dense(inputs, weights, use_bias = use_bias):

		# Load kernel and bias based on use_bias
		kernel = weights[0]
		if use_bias:
			bias = weights[1]

		# Perform matrix multiplication between inputs and kernel
		output = tf.einsum("ij,jk->ik", inputs, kernel)

		# Add bias if use_bias is True
		if use_bias:

			output = tf.nn.bias_add(output, bias)

		# Apply softmax activation function on output
		# because I'm only using the Dense layer as a final classification layer.
		activated_output = tf.keras.activations.softmax(output)

		return activated_output

	return functional_dense