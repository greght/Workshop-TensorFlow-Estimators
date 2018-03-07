# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Based on "A Guide to TF Layers: Building a Convolutional Neural Network" found
# at www.tensorflow.org/versions/r1.4/tutorials/layers

import tensorflow as tf
import numpy as np
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

def model_fn(features, labels, mode, params):
  
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(inputs=input_layer,
                           filters=8,
                           kernel_size=[5, 5],
                           padding="same",
                           activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(inputs=pool1,
                           filters=16,
                           kernel_size=[5, 5],
                           padding="same",
                           activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 16])
  dense = tf.layers.dense(inputs=pool2_flat, units=200, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense,
                              rate=0.4,
                              training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  # If called by 'evaluate' function...
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    
  # Define optimization 
  optimizer = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

  # Else, called by 'train' function
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

if __name__ == '__main__':

    # Load datasets
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    # Reset graph directory
    shutil.rmtree('graphCNN',ignore_errors=True)

    # Get DNN
    model_params = {"learning_rate": 0.1}
    dnn = tf.estimator.Estimator(model_fn=model_fn,
                                 model_dir='graphCNN',
                                 params=model_params)

    # Fit (train) model
    batch_size=100
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=batch_size,
                                                        num_epochs=None,
                                                        shuffle=True)
  
    # Train
    dnn.train(input_fn=train_input_fn, steps=5000)

    # Validate
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                        y=eval_labels,
                                                        batch_size=batch_size,
                                                        num_epochs=1,
                                                        shuffle=False)
    
    accuracy = dnn.evaluate(input_fn=valid_input_fn)["accuracy"]
    print("Accuracy: ", accuracy)
