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
# Modified from the TensorFlow "tf.estimator Quickstart" tutorial at www.tensorflow.org/versions/r1.4/get_started/estimator

import numpy as np
import tensorflow as tf
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

# Read in data
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename="iris_training.csv",
  target_dtype=np.int,
  features_dtype=np.float32)
valid_set = tf.contrib.learn.datasets.base.load_csv_with_header(
  filename="iris_test.csv",
  target_dtype=np.int,
  features_dtype=np.float32)
features = np.array(training_set.data)
labels = np.array(training_set.target)

# Reset graph directory
model_dir = 'graphDNNClassifier'
shutil.rmtree(model_dir,ignore_errors=True)

# Define optimizer
optimizer=tf.train.AdagradOptimizer(learning_rate=0.1)

# Get DNN
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
dnn = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                 hidden_units=[10, 20, 10],
                                 n_classes=3,
                                 model_dir=model_dir,
                                 optimizer=optimizer)
  
# Fit (train) model
batch_size=10
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": features},
                                                    y=labels,
                                                    batch_size=batch_size,
                                                    num_epochs=None,
                                                    shuffle=True)
# Train model
dnn.train(input_fn=train_input_fn, steps=10000)

# Validate
featuresValid = np.array(valid_set.data)
labelsValid = np.array(valid_set.target)
valid_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": featuresValid},
                                                    y=labelsValid,
                                                    batch_size=batch_size,
                                                    num_epochs=1,
                                                    shuffle=False)
accuracy_score = dnn.evaluate(input_fn=valid_input_fn)["accuracy"]
print "Accuracy: ", accuracy_score

# Classify two new flower samples.
new_samples = np.array([[6.4, 3.2, 4.5, 1.5],
                        [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": new_samples},
                                                      num_epochs=1,
                                                      shuffle=False)

predictions = dnn.predict(input_fn=predict_input_fn)
predicted_classes = [p["classes"] for p in predictions]

print "New Samples, Class Predictions: ", predicted_classes
