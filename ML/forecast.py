import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from datetime import datetime, time


def count_tot_seconds(target_date):

    current_time = datetime.now()

    timedelta = current_time - datetime.strptime(target_date,
                                                 '%Y-%m-%d %H:%M:%S')

    return timedelta.days * 24 * 3600 + timedelta.seconds


def forecast(checkpoint_path, data, target_date):

    # Data preprocessing

    data = pd.read_csv(data)

    data = data.values
    data = data[:,2:]
    data = data[:,:-1:]
    
    x_scaler = MinMaxScaler()
    
    # Choosing only last 0.1 times steps as input data for forecasting

    train_split = 0.9
    num_train = int(train_split * len(data))
    num_test = len(data) - num_train

    x_train = data[0:num_train]
    x_test = data[num_train:]

    num_x_signals = x_train.shape[1]
    num_y_signals = num_x_signals


    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    # y_scaler = MinMaxScaler()
    # y_train_scaled = y_scaler.fit_transform(y_train)
    # y_test_scaled = y_scaler.transform(y_test)

    # Load the model
    model = Sequential()
    model.add(
        GRU(units=512,
            return_sequences=True,
            input_shape=(
                None,
                num_x_signals,
            )))

    model.add(Dense(num_y_signals, activation='sigmoid'))

    
    if False:
        from tensorflow.python.keras.initializers import RandomUniform

        init = RandomUniform(minval=-0.05, maxval=0.05)

        # Output later initializer
        model.add(
            Dense(num_y_signals, activation='linear', kernel_initializer=init))

    warmup_steps = 50

    # Load the weights
    # Note that above initialized weights will be ignored

    try:
        model.load_weights(checkpoint_path)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    # PREDICT

    # Start idx should be current time step
    start_idx = 0

    # Count lenght of forecast that corresponds to number of seconds between the currrent time and the target time
    length = count_tot_seconds(target_date)

    # End-index for the sequences.
    end_idx = start_idx + length

    # Input-signals for the model.
    x = x_test_scaled[start_idx:end_idx]
    
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x_test_scaled)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    pred_dict = {i: None for i in target_names}

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        # pred_dict[target_names[signal] = signal_pred

        return pred_dict

pred_dict = forecast('24_checkpoint.keras', 'data.csv', '2018-12-03 00:00:00')

print(pred_dict)