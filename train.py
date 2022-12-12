import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
import pathlib
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from os.path import exists
from google.colab import files

BATCH_SIZE = 2
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_CHANNEL = 3
NUM_CLASSES = 36

def resize_file_data(data):
	data_flattened = data.split(', ')
	for i in range(0, len(data_flattened)):
	data_flattened[i] = int(data_flattened[i])
	# countOfData = len(data_flattened)
	# print(countOfData)
	data = np.resize(data_flattened, (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL))
	# print(data)
	return data

def file_dataset_from_directory(data_path, data_type):
	# Get train, valid and test data from files
	data = []
	label = []
	my_dataset_data = []
  my_dataset_label = []

	for name in class_names:
		# print(name)
		data_dir = data_path / pathlib.Path(name)
		for filename in os.listdir(data_dir):
			file_dir = os.path.join(data_dir, filename)
			# checking if it is a file
			if os.path.isfile(file_dir):
				# print(file_dir)
				with open(file_dir, "r") as f:
					content = f.read()
					# print(file_dir)
					content = resize_file_data(content)
					data.append(content)
					label.append(int(ord(name)))
					# print(data)

	# Shuffle the data and label in the same order
	idx = np.random.permutation(len(data))
	data = np.array(data)
	label = np.array(label)
	data_shuffled, label_shuffled = data[idx], label[idx]
	length = len(label_shuffled)
	# print(label)
	# print(label_shuffled)

	# Batch the data by BATCH_SIZE to format 4D tensor as input data
	dataset_len = int(length / BATCH_SIZE)
	# print(f'dataset_len = {dataset_len}')
	for count in range(dataset_len):
		this_data_batch = [None] * BATCH_SIZE
		this_label_batch = [None] * BATCH_SIZE
		for data_num in range(BATCH_SIZE):
			# print(f"data_num = {data_num}")
			data_index = count * BATCH_SIZE + data_num
			# print(f'data_index = {data_index}')
			this_data_batch[data_num] = data_shuffled[data_index]
			this_label_batch[data_num] = label_shuffled[data_index]

		this_data_batch = tf.convert_to_tensor(this_data_batch)
		this_data_batch = tf.reshape(this_data_batch, [BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
		my_dataset_data.append(this_data_batch)

		this_label_batch = tf.convert_to_tensor(this_label_batch)
		this_label_batch = tf.reshape(this_label_batch, [BATCH_SIZE])
		my_dataset_label.append(this_label_batch)

	print(data_type + "_total_data_length: " + str(length)) 
	return my_dataset_data, my_dataset_label, length

# Define a simple sequential model
def create_model():
	model = tf.keras.Sequential([
		layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL)),
		layers.MaxPooling2D(),
		layers.Dropout(0.1),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Dropout(0.1),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dropout(0.1),
		layers.Dense(NUM_CLASSES, activation="softmax")
	])

	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

	return model

if __name__ == "__main__":
	train_dir = './Users/xue_q/Downloads/keyboard_dataset/train'
	train_dir = pathlib.Path(train_dir)
	val_dir = './Users/xue_q/Downloads/keyboard_dataset/test'
	val_dir = pathlib.Path(val_dir)

	train_input_count = len(list(train_dir.glob('*/*.txt')))
	val_input_count = len(list(val_dir.glob('*/*.txt')))
	# print(f'{train_input_count}, {val_input_count}')

	class_names = np.array([item.name for item in train_dir.glob('*')])
	num_classes = len(class_names)
	if num_classes != NUM_CLASSES:
		print("Warning: Wrong classes number!")
		exit()
	# print(class_names)
	# print(num_classes)

	# Load the train and validation dataset
	train_ds, train_label, train_len = file_dataset_from_directory(train_dir, "train")
	val_ds, val_label, val_len = file_dataset_from_directory(val_dir, "validation")

	# Create a basic model instance
	model = create_model()
	model.summary()

	epochs = 10
	# fit model
	his = model.fit(
		train_ds,
		train_label,
		validation_data = (val_ds, val_label),
		epochs = epochs
	)

	# evaluate model
	_, acc = model.evaluate(val_ds, val_label, verbose=0)
	print('> %.3f' % (acc * 100.0))