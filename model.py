import tensorflow as tf
import numpy as np

initializer = tf.contrib.layers.variance_scaling_initializer()


class DQN:

    def __init__(self, input_size, output_size, name="main"):
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self.build_network()

    def build_network(self, h_size=1000, l_rate=1e-3):

        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First Layer of weights
            layer_1 = tf.layers.dense(inputs=self._X, units=h_size, kernel_initializer=initializer, activation=tf.nn.elu)
            # Second Layer of weights
            self._Qpred = tf.layers.dense(layer_1, units=self.output_size, kernel_initializer=initializer, activation=None)


        self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def set_session(self, sess):
        self.session = sess

    def predict(self, state):
        X = np.reshape(state, newshape=[1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: X})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
