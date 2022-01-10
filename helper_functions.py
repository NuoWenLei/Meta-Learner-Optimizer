import tensorflow as tf
# Function to reshape the weights post processing from meta-learner 
def reshape_weight(cI, learner):
  return [tf.reshape(w, tf.shape(l)) for l, w in zip(learner.trainable_weights, tf.split(cI, [tf.size(i) for i in learner.trainable_weights]))]


# Preprocessing function used in the paper
def preprocess_grad_or_loss(x):
  p = tf.cast(10, tf.float32)

  condition = tf.cast(tf.math.abs(x) >= tf.exp(-1 * p), tf.float32)

  a = (condition * tf.math.log(tf.math.abs(x) + 1e-8) / p) + (1. - condition) * -1.

  b = (((1. - condition) * tf.exp(p)) * x) + (condition) * tf.sign(x)

  return tf.stack((a, b), 1)

def calc_learner_params(l):
  num_learner_params = tf.reduce_sum([tf.size(a) for a in l.trainable_variables]).numpy()
  return num_learner_params

# Create a metric that takes into account the probability aspect of multi-class classification
def accuracy_try(output, target):
  batch_size = tf.shape(target)[0]

  # Find correct predictions
  pred = tf.argmax(output, axis = 1)
  correct = tf.equal(tf.cast(pred, tf.int32), tf.argmax(target, axis = 1, output_type = tf.int32))

  # Calculate percentage accuracy in batch
  return tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(batch_size, tf.float32)

