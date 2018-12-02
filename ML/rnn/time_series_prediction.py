import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


### UTILITY FUNCTIONS FOR DATA PREPROCESSING


class TSP(object):

	def init(self):
		pass

	@staticmethod
	def load_training_data(data,timestamp_col_name):

		# Read the data and convert into pandas dataframe with time stamp as index
		"""Args:
			data = .csv file
			timestamp_col_name = str corresponds to column name holding datetime data

			Returns: pandas df
			"""

		data = pd.read_csv(data,index_col=0)
		data = data.set_index(data.timestamp)

		# Remove the column with the timestamp
		data = data.drop(timestamp_col_name, 1)

		print("Training Data loaded successfully")

		print(data.head())

		return data

	@staticmethod
	def data_choose_training_signals(training_signals):


		"""training_signals = list of column names"""


		data = data[training_signals]

		return data

	@staticmethod
	def convert_df_2_arr(dataframe):

 
		# Remove the index 

		data_arr = dataframe.values
		data_arr= data_arr[:,1:]

		return data_arr

	@staticmethod
	def train_test_split(data_arr,train_split):

		x_data  = data_arr.copy()
		y_data = x_data.copy()  # Target is time series data itself

		# Total number of training samples
		num_train = int(train_split * num_data)

		# Number of testing data
		num_test = num_data - num_train

		# Split the data for training
		x_train = x_data[0:num_train]
		x_test = x_data[num_train:]


		# Targets are same us training data
		y_train = y_data[0:num_train]
		y_test = y_data[num_train:]

	return num_train,num_test,x_train,x_test,y_train,y_test


	@staticmethod
	def get_num_inp_signals(x_data,y_data):


		return x_data.shape[1],y_data.shape[1]

	@staticmethod
	def scale_data(x_train,x_test,y_train,y_test):


		scaler = MinMaxScaler()

		# Scaling inputs
		x_train_scaled = scaler.fit_transform(x_train)
		x_test_scaled = scaler.transform(x_test)

		# Scale targets
		y_train_scaled = y_scaler.fit_transform(y_train)
		y_test_scaled = y_scaler.transform(y_test)

		return x_train_scaled, x_test_scaled,y_train_scaled,y_test_scaled



	# DATA GENERATION FOR TRAINING AND TESTING

	@staticmethod
	def batch_generator(batch_size, sequence_length):
	    """
	    Generator function for creating random batches of training-data.
	    """

	    # Infinite loop.
	    while True:
	        # Allocate a new array for the batch of input-signals.
	        x_shape = (batch_size, sequence_length, num_x_signals)
	        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

	        # Allocate a new array for the batch of output-signals.
	        y_shape = (batch_size, sequence_length, num_y_signals)
	        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

	        # Fill the batch with random sequences of data.
	        for i in range(batch_size):
	            # Get a random start-index.
	            # This points somewhere into the training-data.
	            idx = np.random.randint(num_train - sequence_length)
	            
	            # Copy the sequences of data starting at this index.
	            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
	            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
	        
	        yield (x_batch, y_batch)


	@staticmethod
	def plot_batch(x_batch,batch,signal):

		##### Plot the input batch and signal 

		batch = 0   # First sequence in the batch.
		signal = 0  # First signal from the 3 input-signals (X,Y,Z).
		seq = x_batch[batch, :, signal]
		plt.plot(seq)

		return plt.show()


	@staticmethod
	def get_validation_data(x_test,y_test):

		validation_data = (np.expand_dims(x_test_scaled, axis=0),
	    	               np.expand_dims(y_test_scaled, axis=0))

		return validation_data

	@staticmethod
	def build_model(num_x_signals,num_y_signals,num_units,return_sequences):


		# num_x_signals - Int Size of input signals
		# num_y_signals - Int Size of target signals
		# num_units - Size of recurrent units
		# return_sequences - Boolean, Stack LSTM layers 
		# activation - 

		model = Sequential()

		model.add(GRU(units=num_units,
	              return_sequences=return_sequences,
	              input_shape=(None, num_x_signals)))

		model.add(Dense(num_y_signals, activation=activation))

		# Initialize weights

		if False:
	    from tensorflow.python.keras.initializers import RandomUniform

	    # Maybe use lower init-ranges.
	    init = RandomUniform(minval=-0.05, maxval=0.05)

	    # Densely connected output layer with linear output units
	    model.add(Dense(num_y_signals,
	                    activation='linear',
	                    kernel_initializer=init))

	    return model


	# LOSS FUNCTION

	@staticmethod
	def loss_mse_warmup(y_true, y_pred,warmup_steps):
	    """
	    Calculate the Mean Squared Error between y_true and y_pred,
	    but ignore the beginning "warmup" part of the sequences.
	    
	    y_true is the desired output.
	    y_pred is the model's output.
	    """

	    # The shape of both input tensors are:
	    # [batch_size, sequence_length, num_y_signals].

	    # Ignore the "warmup" parts of the sequences
	    # by taking slices of the tensors.
	    y_true_slice = y_true[:, warmup_steps:, :]
	    y_pred_slice = y_pred[:, warmup_steps:, :]

	    # These sliced tensors both have this shape:
	    # [batch_size, sequence_length - warmup_steps, num_y_signals]

	    # Calculate the MSE loss for each value in these tensors.
	    # This outputs a 3-rank tensor of the same shape.
	    loss = tf.losses.mean_squared_error(labels=y_true_slice,
	                                        predictions=y_pred_slice)

	    # Keras may reduce this across the first axis (the batch)
	    # but the semantics are unclear, so to be sure we use
	    # the loss across the entire tensor, we reduce it to a
	    # single scalar with the mean function.
	    loss_mean = tf.reduce_mean(loss)

	    return loss_mean

	@staticmethod
	def train_model(model,optimizer,lr,loss):

		if optimizer == 'RMSprop':
			optimizer = RMSprop(lr=1e-3)

		# loss is loss_mse_warmup 
		model.compile(loss=loss, optimizer=optimizer) 

		return model

	# VISUALIZING THE MODEL GRAPH

	@staticmethod
	def save_graph(session, keep_var_names=None, output_names=None, clear_devices=True):
	    """
	    Freezes the state of a session into a pruned computation graph.

	    Creates a new computation graph where variable nodes are replaced by
	    constants taking their current value in the session. The new graph will be
	    pruned so subgraphs that are not necessary to compute the requested
	    outputs are removed.
	    @param session The TensorFlow session to be frozen.
	    @param keep_var_names A list of variable names that should not be frozen,
	                          or None to freeze all the variables in the graph.
	    @param output_names Names of the relevant graph outputs.
	    @param clear_devices Remove the device directives from the graph for better portability.
	    @return The frozen graph definition.
	    """
	    
	    #### Load the pb file using tensorflow
		from tensorflow.keras import backend as K

	    graph = session.graph
	    with graph.as_default():
	        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
	        output_names = output_names or []
	        output_names += [v.op.name for v in tf.global_variables()]
	        input_graph_def = graph.as_graph_def()
	        if clear_devices:
	            for node in input_graph_def.node:
	                node.device = ""
	        frozen_graph = tf.graph_util.convert_variables_to_constants(
	            session, input_graph_def, output_names, freeze_var_names)


	    #### Save keras model as tf pb file

		frozen_graph = freeze_session(K.get_session(),output_names = [out.op.name for out in model.outputs])
		return tf.train.write_graph(frozen_graph,os.getcwd(),'modelgraph',as_text= False)


	@staticmethod
	def evaluate_model(x_test_scaled,y_test_scaled, path_checkpoint,model):

		try:
	    	model.load_weights(path_checkpoint)
		except Exception as error:
	    	print("Error trying to load checkpoint.")
	    	print(error)

	    result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
	                        y=np.expand_dims(y_test_scaled, axis=0))

	    return result



	@staticmethod
	def plot_prediction(start_idx, length=100, train=True, forecast = False):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    if forecast:
        x = x_test_scaled.copy()

    elif train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.

    if forecast:
        x = x[start_idx:end_idx]
    else:
        
    	x = x[start_idx:end_idx]
    	y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    pred_dict = {i: None for i in target_names}
    # For each output-signal.
    for signal in range(len(target_names)):
        
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        if forecast = False:
        	
        	# Get the true output-signal from the data-set.
        	signal_true = y_true[:, signal]
        
        pred_dict[target_names[signal]] = signal_pred

	    # Make the plotting-canvas bigger.
	    plt.figure(figsize=(15,5))
	        
	    # Plot and compare the two signals.
	    plt.plot(signal_true, label='true')
	    plt.plot(signal_pred, label='pred')
	        
	    # Plot grey box for warmup-period.
	    p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
	        
	    # Plot labels etc.
	    plt.ylabel(target_names[signal])
	    plt.legend()
	    plt.show()

	def on_click_predict(start_idx, predict_length, is_training, x_train_scaled, y_train, plot)
    
	    plot_prediction(start_idx=start_idx, length=predict_length, train=is_training)

	def on_click_forcasting(start_idx, predict_length, is_training)
    
    	plot_prediction(start_idx=start_idx, length=predict_length, train=is_training)

if __name__ == __main__:


	# Init TSP instance 
	tsp = TSP()

	# GET INPUTS FROM GUI
	
	data = 'Load the .csv file'
	timestamp_col_name = 'datatime column name'
	training_signals = ['x','y','z']
	batch_size = 128
	sequence_length = 900  
	warmup_steps = 50
	epochs = 

	steps_per_epoch = 

	optimizer = 

	show_model_summary = 
	# Get the data

	data = tsp.load_training_data(data,timestamp_col_name):

	list_available_training_signals = list(data)

	dataframe = tsp.data_choose_training_signals(training_signals)

	data_arr = tsp.convert_df_2_arr(dataframe)

	num_train,num_test,x_train,x_test,y_train,y_test = train_test_split(data_arr,train_split):

	num_x_signals, num_y_signals= get_num_inp_signals(x_data,y_data)

	x_train_scaled, x_test_scaled,y_train_scaled,y_test_scaled = scale_data(x_train,x_test,y_train,y_test)


	# Batch data generator
	generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)


	# Validation data
	validation_data = tsp.get_validation_data(x_test_scaled,y_test_scaled)


	# Prepare the model
	model = build_model(num_x_signals,num_y_signals,num_units,return_sequences)


	if tsp.optimizer == 'RMSprop':
			optimizer = RMSprop(lr=1e-3)

	loss_func = loss_mse_warmup(y_true, y_pred):
	
	model = model.compile(loss=loss_func, optimizer=optimizer) 

	if show_model_summary:
		model.summary()


	# Model checkpoint saver

	path_checkpoint = 'training_checkpoint.keras'
	callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

	# Overfitting check
	callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

	# Save training logs 
	callback_tensorboard = TensorBoard(log_dir='./training_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
	
	# Update size of learning rate
	callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

	# List of callbacks

	callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


    # On click of Start training

    %%time

	model.fit_generator(generator=generator,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_data,
                    callbacks=callbacks)


	# Onclick of Evaluate model

	try:
    	model.load_weights(path_checkpoint)
	except Exception as error:
    	print("Error trying to load checkpoint.")
    	print(error)


    result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


    print("loss (test-set):", result)


    # On click of Forcasting

    start_idx = 0
    predict_length = 1000
    is_training = False

    if plot_prediction:
    	plot_prediction(start_idx=start_idx, length=predict_length, train=is_training)