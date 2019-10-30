"""
The file to initialize the training of the models. For the given structure of the project.
"""

import tensorflow as tf
from tensorflow import keras 
import numpy as np
import pandas as pd
import pickle 

def get_lstm_model(batch_input_shape = (1,6,1)):
    """
    returns an uncompiled lstm model for the wave dataset
    """
    model = keras.models.Sequential([
        keras.layers.LSTM(
            16, 
            stateful = True, 
            batch_input_shape = batch_input_shape, 
            implementation = 1),
        keras.layers.Dense(1, activation = keras.activations.softplus, use_bias=False)
        ])

    return model

def get_parallel_pred_model(batch_input_shape = (1,6,1)):
    """
    unused
    """
    model = keras.models.Sequential([
        keras.layers.LSTM(
            128, 
            stateful = True,
            batch_input_shape = batch_input_shape,
            ),
        keras.layers.Dense(6,activation = keras.activations.softplus)
        ])
    return model

def data(output_col):
    """
    returns a tuple of np.arrays for the training.
    """
    path = "../data/buoy_data.csv"
    index_col = "Date/Time"
    n_col = 6

    df = pd.read_csv(path)
    df.set_index(index_col, inplace=True)
    df = df.loc['01/01/2017 01:00':]
    df.rename(columns={"Peak Direction": "Peak_Direction"}, inplace=True)
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
            'Peak_Direction', 
            'SST'
            ]

    tf.random.set_seed(11)

    for col in output_cols:

        print(f"STARTING TRAINING FOR MODEL {col}")

        x_arr, y_arr = data(col)
        
        opt = keras.optimizers.Adam(lr = 0.00001)
        if col == "Hs" or col == "Hmax" or col == "SST":
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
        # remove this line if files do not exist
        model.load_weights(f"../models/lstm_{col}.h5")

        if col == "SST":
            opt = keras.optimizers.Adam(lr = 0.0000001)

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
        hist_df.to_pickle(f"../models/lstm_{col}_hist.pkl")
