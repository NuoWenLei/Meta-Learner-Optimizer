import tensorflow as tf

# Special Meta LSTM cell that is very similar to normal LSTM cells but uses different weights
class MetaLSTMCell(tf.keras.layers.Layer):
	def __init__(self, args, learner_param_size):
		super(MetaLSTMCell, self).__init__()

		self.input_size = args["input_size"]
		self.hidden_size = args["hidden_size"]
		self.learner_param_size = learner_param_size

		# Weight initializers
		self.uniform_initializer = tf.random_uniform_initializer(minval = -0.01, maxval = 0.01)

		# Bias initializers
		# I had kept the biases initialized at 0 for the LONGEST TIME and that's what I didn't realize was wrong.
		# This is what got me stuck for a week because I didn't realize that
		# if the meta-learner didn't make big changes to the gradient in the beginning (due to small weights and biases),
		# the meta-learner just wouldn't do gradient descent because all the weights are so small.
		self.uniform_initializer_bI = tf.random_uniform_initializer(minval = args["bI_init_1"], maxval = args["bI_init_0"])
		self.uniform_initializer_bF = tf.random_uniform_initializer(minval = args["bF_init_0"], maxval = args["bF_init_1"])

		# Initialization of weights/kernels and biases of the meta-learner
		self.W_i = tf.Variable(self.uniform_initializer(shape = [self.input_size + 2, 1], dtype = tf.float32))
		self.W_f = tf.Variable(self.uniform_initializer(shape = [self.input_size + 2, 1], dtype = tf.float32))

		self.b_i = tf.Variable(self.uniform_initializer_bI(shape = (1, 1), dtype = tf.float32))
		self.b_f = tf.Variable(self.uniform_initializer_bF(shape = (1, 1), dtype = tf.float32))


		# Attempt at using Dense layers to do wide processing on loss and gradients

		# self.dense_f = Dense(hidden_size, activation = "relu")
		# self.congregate_dense_f = Dense(1, activation = "sigmoid")
		# self.dense_i = Dense(hidden_size, activation = "relu")
		# self.congregate_dense_i = Dense(1, activation = "sigmoid")
		# self.c_i = self.uniform_initializer(shape = (learner_param_size, 1), dtype = tf.float32)


	def call(self, inputs, hidden_states = None, c_i = None):
		"""
		INPUTS:
			inputs:
			- x_inputs (latest short term memory of loss and gradient)
				shape: (learner_param_size, input_size)
			- new_gradients (gradient of learner model)
				shape: (learner_param_size)
			
			hidden_states:
			- prev_f (previous forget gate)
				shape: (learner_param_size, 1)
			- prev_i (previous input gate)
				shape: (learner_param_size, 1)
			- prev_c (previous cell state/long term memory)
				shape: (learner_param_size, 1)

			c_i:
			flattened weights of the learner.

			Only used if hidden_states == None
			because it is a new learner and there is no record
			of previous weights.

		OUTPUTS:
			- new_c (new cell state, which also functions as the weights for the learner)
			- hidden_states (new hidden states, which will be used in the next iteration)
		"""

		x_inputs, new_gradients = inputs

		if hidden_states == None:

			# Initialize hidden states
			# if starting training with a new learner
			prev_f = tf.zeros((self.learner_param_size, 1))
			prev_i = tf.zeros((self.learner_param_size, 1))
			prev_c = tf.reshape(c_i, (-1,1))

			hidden_states = [prev_f, prev_i, prev_c]

		prev_f, prev_i, prev_c = hidden_states


		# Apply kernels and biases to inputs in a way VERY SIMILAR to LSTMs.
		# The main difference being that we only want to calculate the cell state due to its stability.
		# The reason there are only 3 inputs instead of 4 is because x_inputs count as both the loss and gradients
		new_f = tf.matmul(tf.concat([x_inputs, prev_c, prev_f], axis = 1), self.W_f) + self.b_f

		new_i = tf.matmul(tf.concat([x_inputs, prev_c, prev_i], axis = 1), self.W_i) + self.b_i

		new_c = (tf.math.sigmoid(new_f) * prev_c) - (tf.math.sigmoid(new_i) * tf.reshape(new_gradients, (-1, 1)))

		# Attempt at using Dense layers

		# new_f_hidden = self.dense_f(tf.concat([x_inputs, prev_c, prev_f], axis = 1))

		# new_i_hidden = self.dense_i(tf.concat([x_inputs, prev_c, prev_i], axis = 1))

		# new_f_dropout = self.dropout_f(new_f_hidden)

		# new_i_dropout = self.dropout_i(new_i_hidden)

		# new_f = self.congregate_dense_f(new_f_hidden)

		# new_i = self.congregate_dense_i(new_i_hidden)

		# new_c = (new_f * prev_c) - (new_i * tf.reshape(new_gradients, (-1,1)))

		# new_f = tf.reshape(
		#     tf.reduce_sum(
		#         tf.einsum("ij,jk->ik",
		#                   tf.concat([x_inputs, prev_c, prev_f], axis = 1), self.W_f) + self.b_f, axis = -1), (-1,1))

		# new_i = tf.reshape(
		#     tf.reduce_sum(
		#         tf.einsum("ij,jk->ik",
		#                   tf.concat([x_inputs, prev_c, prev_i], axis = 1), self.W_i) + self.b_i, axis = -1), (-1,1))

		return new_c, [new_f, new_i, new_c]


# Overall MetaLearner Class
class MetaLearner(tf.keras.models.Model):
	def __init__(self, args, learner_param_size):
		super(MetaLearner, self).__init__()

		self.input_size = args["input_size"]
		self.hidden_size = args["hidden_size"]
		self.learner_param_size = learner_param_size

		# Create LSTM cell and Meta LSTM cell with appropriate parameters
		self.lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size, activation = "relu")
		self.meta_lstm = MetaLSTMCell(args, learner_param_size)

	def call(self, inputs, hidden_states = None, c_i = None):
		"""
		INPUTS:
		inputs:
		- loss (most recent loss of learner model)
			shape: (learner_param_size, 2)
		- preprocessed_gradients (preprocessed gradients of learner model)
			shape: (learner_param_size, 2)
		- raw_gradients (gradients of the learner model)
			shape: (learner_param_size,)

		hidden_states:
		- hidden states for LSTM Cell
		- hidden states for Meta LSTM Cell

		c_i:
		flattened weights of the learner.

		Only used if hidden_states == None
		because it is a new learner and there is no record
		of previous weights.

		OUTPUTS:
		- new_c (new cell state, which also functions as the weights for the learner)
		- hidden_states (new hidden states, which will be used in the next iteration)
		"""

		loss, preprocessed_gradients, raw_gradients = inputs

		# Expand loss shape to concatenate it with the gradient,
		# resulting in a shape of (NUM_LEARNER_PARAMS, 4).
		#
		# The reason there are 4 features is due to the preprocessing function
		# done on the loss and gradients, making them each of shape (n, 2).
		#
		# You can think of it as the meta-learner processing an input
		# with a batch size of NUM_LEARNER_PARAMS, each row with 4 features.
		loss = tf.repeat(loss, tf.shape(preprocessed_gradients)[0], axis = 0)
		inputs = tf.concat([loss, preprocessed_gradients], axis = 1)

		if hidden_states == None:
			# Initialize hidden states for the normal LSTM cell
			# if it is a new learner.
			h = tf.zeros((self.learner_param_size, self.hidden_size), dtype = tf.float32)
			c = tf.zeros((self.learner_param_size, self.hidden_size), dtype = tf.float32)
			hidden_states = [(h, c), None]

		# Process loss and gradient with LSTM Cell
		# and then pass the hidden vector, or short term memory,
		# with the raw gradients into the Meta LSTM Cell
		_, (hidden_x, cell_x) = self.lstm_cell(inputs, hidden_states[0])

		new_gradients, meta_hidden_states = self.meta_lstm([hidden_x, raw_gradients], hidden_states[1], c_i = c_i)

		return tf.squeeze(new_gradients), [(hidden_x, cell_x), meta_hidden_states]
