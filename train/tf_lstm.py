"""

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

class LSTM_model(keras.Model):

    def __init__(self, input_size, output_size):
        super(LSTM_model, self).__init__()
        
        self.pred = tf.random.normal(shape=(output_size, 1))
        self.hidden_state = tf.random.normal(shape=(output_size, 1))

        self.forget_x = tf.random.normal(shape=(output_size, input_size))
        self.forget_h = tf.random.normal(shape=(output_size, output_size))
        self.forget_b = tf.random.normal(shape = (output_size,1))
        
        self.input_x = tf.random.normal(shape=(output_size, input_size))
        self.input_h = tf.random.normal(shape=(output_size, output_size))
        self.input_b = tf.random.normal(shape = (output_size,1))

        self.b_input_x = tf.random.normal(shape=(output_size, input_size))
        self.b_input_h = tf.random.normal(shape=(output_size, output_size))
        self.b_input_b = tf.random.normal(shape = (output_size,1))

        self.output_x = tf.random.normal(shape=(output_size, input_size))
        self.output_h = tf.random.normal(shape=(output_size, output_size))
        self.output_b = tf.random.normal(shape = (output_size,1))

    def call(self, x):
        fgate = tf.math.sigmoid(
                tf.matmul(self.forget_x, x) + 
                tf.matmul(self.forget_h, self.pred) + 
                self.forget_b)

        igate = tf.math.tanh(
                tf.matmul(self.input_x, x) + 
                tf.matmul(self.input_h, self.pred) +
                self.input_b) 
        
        bigate = tf.math.sigmoid(
                tf.matmul(self.b_input_x, x) + 
                tf.matmul(self.b_input_h, self.pred) + 
                self.b_input_b)

        igate *= bigate

        ogate = tf.math.sigmoid(
                tf.matmul(self.output_x, x) + 
                tf.matmul(self.output_h, self.pred) + 
                self.output_b)
        
        self.hidden_state = self.hidden_state * fgate + igate
        self.pred = tf.math.tanh(self.hidden_state) * ogate

        return self.pred


if __name__ == "__main__":
    x = np.random.normal(0,1, (100,10))
    y = np.sum(x, axis = 1)
    y = y / np.max(np.abs(y))

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    model = LSTM_model(10, 1)
    model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.MSE)

    model.fit(x,y)

    
