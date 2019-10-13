"""
THERE ARE MISSING VALUES SET TO -99 !
"""

import tensorflow as tf
from tensorflow import keras 
import numpy as np
import pandas as pd
import pickle 

def get_lstm_model(batch_input_shape = (1,6,1)):
    model = keras.models.Sequential([
        keras.layers.LSTM(
            16, 
            stateful = True, 
            batch_input_shape = batch_input_shape, 
            implementation = 1),
        keras.layers.Dense(1, activation = keras.activations.softplus, use_bias=False)
        ])

    return model

def data(output_col):
    path = "../data/buoy_data.csv"
    index_col = "Date/Time"
    n_col = 6

    df = pd.read_csv(path)
    df.set_index(index_col, inplace=True)
    df = df.loc['01/01/2017 01:00':]
    df.interpolate(inplace=True)
    x_arr = df[:-1].values
    y_arr = df[output_col][1:].values

    x_arr = np.reshape(x_arr, (-1, n_col, 1))
    return x_arr, y_arr

if __name__ == "__main__":

    output_cols = [
            'Hs', 
            'Hmax', 
            'Tz', 
            'Tp', 
            'Peak Direction', 
            'SST'
            ]

    tf.random.set_seed(11)

    for col in output_cols:

        print(f"STARTING TRAINING FOR MODEL {col}")

        x_arr, y_arr = data(col)
        
        opt = keras.optimizers.Adam(lr = 0.00008)
        if col == "Hs" or col == "Hmax":
            loss = keras.losses.MAE
        else:
            loss = keras.losses.MSE

        metrics = [
                keras.losses.MSE,
                keras.losses.MAE,
                keras.metrics.KLD
                ] 

        callbacks = [
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
                ]


        model = get_lstm_model()
        model.compile(
                optimizer=opt,
                loss=loss,
                metrics=metrics
                )
        
        hist = model.fit(
                x_arr,
                y_arr,
                epochs = 150,
                batch_size = 1,
                shuffle = False,
                callbacks = callbacks,
                validation_split = 0.2,
                use_multiprocessing = True,
                workers = 4
                )

        model.save_weights(f"../models/lstm_{col}.h5") 
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_pickle("../models/lstm_{col}_hist.pkl")
