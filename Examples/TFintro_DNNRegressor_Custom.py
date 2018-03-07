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

# Modified from  "A Guide to TF Layers: Building a Convolutional Neural Network" found
# at www.tensorflow.org/versions/r1.4/tutorials/layers

import tensorflow as tf
import numpy as np
import shutil

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

tf.logging.set_verbosity(tf.logging.INFO)

def model_fn(features, labels, mode, params):

    # Define DNN
    Layers = [2,50,130,25,1]
    a1 = tf.layers.dense(inputs=features["x"],units=Layers[1],activation=tf.nn.relu)
    a2 = tf.layers.dense(inputs=a1,units=Layers[2],activation=tf.nn.relu)
    a3 = tf.layers.dense(inputs=a2,units=Layers[3],activation=tf.nn.relu)
    f = tf.layers.dense(inputs=a3,units=Layers[4],activation=None)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(f, [-1])
    
    # If called by 'predict' function...
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predictions)
    
    # If called by 'evaluate' function...
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    # Define optimization
    optimizer = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # Else, called by 'train' function
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

if __name__ == '__main__':

    # Read in data
    dataIn = np.genfromtxt('dataRegression_train.csv',delimiter=',')
    features = dataIn[:,0:2]
    labels = dataIn[:,2]
    
    # Reset graph directory
    model_dir = 'graphDNNRegressor'
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
    
    # Train
    dnn.train(input_fn=train_input_fn, steps=10000)

    # Validate
    dataInValid = np.genfromtxt('dataRegression_valid.csv',delimiter=',')
    featuresValid = dataIn[:,0:2]
    labelsValid = dataIn[:,2]
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": featuresValid},
                                                        y=labelsValid,
                                                        batch_size=batch_size,
                                                        num_epochs=1,
                                                        shuffle=False)
    loss = dnn.evaluate(input_fn=valid_input_fn)["loss"]
    print(loss)

    # Create a prediction set
    x_min = np.amin(features,axis=0)
    x_max = np.amax(features,axis=0)
    x_predict = np.mgrid[x_min[0]:x_max[0]:25j, x_min[1]:x_max[1]:25j].reshape(2,-1).T

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_predict},
                                                          num_epochs=1,
                                                          shuffle=False)
    predictions = dnn.predict(input_fn=predict_input_fn);
    y_predict = np.array([p['predictions'] for p in predictions])

    # Plot the actual and predicted values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x1 = x_predict[:,0].reshape(25,-1)
    x2 = x_predict[:,1].reshape(25,-1)
    y1 = y_predict.reshape(25,-1)
    ax.scatter(features[:,0], features[:,1], labels, c='r', marker='o', label='Actual')
    ax.plot_surface(x1,x2,y1,cmap=cm.coolwarm,linewidth=0,rstride=1,cstride=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')

    plt.show()

