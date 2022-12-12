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
input_height = 32
input_width = 32
input_channel = 3


def resize_file_data(data):
	data_flattened = data.split(', ')
	for i in range(0, len(data_flattened)):
	data_flattened[i] = int(data_flattened[i])
	# countOfData = len(data_flattened)
	# print(countOfData)
	data = np.resize(data_flattened, (input_height, input_width, input_channel))
	# print(data)
	return data

def file_dataset_from_directory(data_path, data_type):
	# Get train, valid and test data from files
	data = []
	label = []

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
				label.append(name)
				# print(data)

	# Shuffle the data and label in the same order
	idx = np.random.permutation(len(data))
	data = np.array(data)
	label = np.array(label)
	data_shuffled, label_shuffled = data[idx], label[idx]
	length = len(label_shuffled)
	# print(label)
	# print(label_shuffled)

	# TODO: Batch the data by batch_size
	data_shuffled = tf.convert_to_tensor(data_shuffled, dtype=tf.float32)
	print(data_shuffled)

	label_shuffled = tf.convert_to_tensor(label_shuffled)
	print(label_shuffled)

	print(data_type + "_data_length:" + str(length))   
	return data_shuffled, label_shuffled, length


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
	# print(class_names)
	# print(num_classes)

	# Load the train and validation dataset
	train_ds, train_label, train_len = file_dataset_from_directory(train_dir, "train")
	val_ds, val_label, val_len = file_dataset_from_directory(val_dir, "validation")

	# Create a basic model instance
	model = create_model()
	model.summary()

	epochs=10
		history = model.fit(
	 	train_ds,
		validation_data = val_ds,
		epochs = epochs
	)