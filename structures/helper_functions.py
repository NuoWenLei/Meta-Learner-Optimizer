import tensorflow as tf
# Function to reshape the weights post processing from meta-learner 
def reshape_weight(cI, learner):
  return [tf.reshape(w, tf.shape(l)) for l, w in zip(learner.trainable_weights, tf.split(cI, [tf.size(i) for i in learner.trainable_weights]))]