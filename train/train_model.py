"""

"""

import torch
import torch.multiprocessing as mp
import pandas as pd 
import numpy as np

import torch_lstm as lstm

def get_train_test_arrays(path):
    """

    """
    df = pd.read_csv(path)
    df.set_index("Date/Time", inplace = True)
    df_train = df[:30000]
    df_test = df[30000:]

    def gen_x_train():
        arr = df_train[:-1].values
        for row in arr:
            yield torch.Tensor(row)

    def gen_y_train():
        arr = df_train[1:].values
        for row in arr:
            yield torch.Tensor(row)

    def gen_x_test():
        arr = df_test[:-1].values
        for row in arr:
            yield torch.Tensor(row)

    def gen_y_test():
        arr = df_test[1:].values
        for row in arr:
            yield torch.Tensor(row)

    return (gen_x_train(), gen_y_train()), (gen_x_test(), gen_y_test())

def main():
    """

    """
    inputfile = './data/buoy_data.csv'
    outputfile = './models/lstm_1.pkl'
   
    (train_x_gen, train_y_gen), (test_x_gen, test_y_gen) = get_train_test_arrays(inputfile)

    num_processes = 4
    model = lstm.LSTM(6,6)
    model.share_memory()

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=lstm.train_gen, args=(model,train_x_gen, train_y_gen, test_x_gen, test_y_gen, 2.5))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
                                                    
    torch.save(model, outputfile)

if __name__ == "__main__":
    main()
   
