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
# Modified from  "A Guide to TF Layers: Building a Convolutional Neural Network" found
# at www.tensorflow.org/versions/r1.4/tutorials/layers

import tensorflow as tf
import numpy as np
import shutil, os
from six.moves.urllib.request import urlopen

tf.logging.set_verbosity(tf.logging.INFO)

def model_fn(features, labels, mode, params):

    # Define DNN
    Layers = [2,10,20,10,3]
    a1 = tf.layers.dense(inputs=features["x"],units=Layers[1],activation=tf.nn.relu)
    a2 = tf.layers.dense(inputs=a1,units=Layers[2],activation=tf.nn.relu)
    a3 = tf.layers.dense(inputs=a2,units=Layers[3],activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=a3,units=Layers[4],activation=None)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    # If called by 'predict' function...
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
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

    # Read in data, download first if necessary
    if not os.path.exists("iris_training.csv"):
        raw = urlopen("http://download.tensorflow.org/data/iris_training.csv").read()
        with open("iris_training.csv", "wb") as f:
            f.write(raw)
      
    if not os.path.exists("iris_test.csv"):
        raw = urlopen("http://download.tensorflow.org/data/iris_test.csv").read()
        with open("iris_test.csv", "wb") as f:
            f.write(raw)
            
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
    model_dir = 'graphDNNClassifier_Custom'
    shutil.rmtree(model_dir,ignore_errors=True)

    # Get DNN
    model_params = {"learning_rate": 0.1}
    dnn = tf.estimator.Estimator(model_fn=model_fn,
                                 model_dir=model_dir,
                                 params=model_params)

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
    print("Accuracy: ", accuracy_score)

    # Classify two new flower samples.
    x_predict = np.array([[6.4, 3.2, 4.5, 1.5],
                          [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_predict},
                                                          num_epochs=1,
                                                          shuffle=False)

    predictions = dnn.predict(input_fn=predict_input_fn)
    predicted_classes = [p["classes"] for p in predictions]
    
    print("New Samples, Class Predictions: ", predicted_classes)
