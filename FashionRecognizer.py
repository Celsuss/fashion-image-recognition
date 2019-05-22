"""
Using Tensorflow 2.0.0-alpha0

Labels

Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot 
"""

#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import sys

print(tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

#%%
DRAW = False

###################
# Hyperparameters #
###################
learning_rate = 0.001

height, width, channels = 28, 28, 1
n_epochs = 30
batch_Size = 64
batch_count = 0

checkpoint_path = './tmp'

#%%
#################
# Draw function #
#################
def drawImage(data):
    if DRAW:
        image = data.reshape(width, height) 
        plt.imshow(image, cmap=plt.cm.binary)
        plt.show()


#%%
############################
# Load and preprocess data #
############################
def loadData(path):
    X_data_train = pd.read_csv(path+"fashion-mnist_train.csv")
    y_data_train = X_data_train.pop("label")
    
    X_data_test = pd.read_csv(path+"fashion-mnist_test.csv")
    y_data_test = X_data_test.pop("label")
    
    print('X_data_train:\n {}'.format(X_data_train.describe()))
    print(X_data_train.head())
    # print(y_data_train.describe())
    
    X_train = np.array(X_data_train)
    y_train = np.array(y_data_train)
    
    X_test = np.array(X_data_test)
    y_test = np.array(y_data_test)
    
    print('Train samples: {}, Test samples: {}'.format(X_data_train.shape[0], X_data_test.shape[0]))
    
    n_test_and_val = int(X_test.shape[0]/2)
    X_val = X_test[:n_test_and_val]
    y_val = y_test[:n_test_and_val]
    X_test = X_test[n_test_and_val:]
    y_test = y_test[n_test_and_val:]
    
    print('Validation samples: {}, Test samples: {}'.format(X_val.shape[0], X_test.shape[0]))
    return X_train, y_train, X_val, y_val, X_test, y_test

def reshapeTrainingData(data):
    data = data.reshape(data.shape[0], height, width, channels)
    return data

def makeOneHot(labels):
    labels = tf.one_hot(labels, labels.max()+1)
    return labels

# TODO: Implment preprocessing
def preprocessData(image_raw):
    image = tf.cast(image_raw, tf.float32)
    image = image / 255.0
    return image

def createDataset(X, y):
    global batch_count
    size = len(X)

    X_dataset = tf.data.Dataset.from_tensor_slices(X)
    y_dataset = tf.data.Dataset.from_tensor_slices(y)

    X_dataset = X_dataset.map(preprocessData, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((X_dataset, y_dataset))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_Size)
    if batch_count == 0:
        batch_count = math.ceil(size/batch_Size)
    print('Created dataset: {}'.format(dataset))
    return dataset

#%%
########################
# Checkpoint functions #
########################
def createCheckpoint(model, optimizer):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckptManager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=2)
    return checkpoint, ckptManager

def restoreModel(checkpoint, ckptManager):
    checkpoint.restore(ckptManager.latest_checkpoint)
    if ckptManager.latest_checkpoint:
        print('Restored checkpoint from {} at epoch {}'.format(ckptManager.latest_checkpoint, start_epoch))
    else:
        print('No checkpoints found, start training at epoch 0')

#%%
#############################
# Create the neural network #
#############################
def createNetwork(n_inputs, n_outputs):
    filters = [32, 64, 64]
    kernel_size = [3,3]
    pool_size = (2, 2)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Convolution2D(filters[0], kernel_size, activation=tf.nn.relu, input_shape=n_inputs))

    for f in filters[1:-1]:
        model.add(tf.keras.layers.Convolution2D(f, kernel_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size))
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Convolution2D(filters[-1], kernel_size, activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(n_outputs, activation=tf.nn.softmax))

    return model

def trainNetwork(model, training_data, validation_data):
    global batch_count
    # loss_op = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_op = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='validation_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')

    # Create a Checkpoint and a CheckpointManager
    checkpoint, ckptManager = createCheckpoint(model, optimizer)
    # Restore the model if there is any checkpoint.
    restoreModel(checkpoint, ckptManager)

    print('**** Start training ****')
    
    for epoch in range(n_epochs):
        startTime = time.time()

        n_batch = 0
        for image_batch, label_batch in training_data:
            n_batch += 1
            loss = trainStep(image_batch, label_batch, model, optimizer, loss_op, train_loss, train_accuracy)

            print("Epoch {}, Batch {}/{}, Train loss {}, Train accuracy {}".format(
                epoch+1, n_batch, batch_count, train_loss.result(), train_accuracy.result()*100), end='\r')

        # saveCheckpointAndImage(ckptManager, generator, seed, epoch)
        print('')
        print('Time for epoch {} is {} sec\n'.format(epoch+1, time.time()-startTime))
                

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def trainStep(images, labels, model, optimizer, loss_op, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_op(predictions, labels)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

    return loss

@tf.function
def validationStep(images, labels, model, loss_op, validation_loss, validation_accuracy):
    predictions = model(images)
    loss = loss_op(predictions, labels)
    validation_loss(loss)
    validation_accuracy(labels, predictions)

#%%
if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '-d':
            DRAW = True

    print("** Fashion Recognizer **")
    X_train, y_train, X_val, y_val, X_test, y_test = loadData('Data/')
    y_train = makeOneHot(y_train)
    y_val = makeOneHot(y_val)
    y_test = makeOneHot(y_test)

    X_train = reshapeTrainingData(X_train)
    X_val = reshapeTrainingData(X_val)
    X_test = reshapeTrainingData(X_test)

    drawImage(X_train[10])

    input_shape = X_train.shape[1:]     # (28, 28, 1)
    n_output = y_train.shape[-1]        # 10

    training_dataset = createDataset(X_train, y_train)
    validation_dataset = createDataset(X_val, y_val)

    model = createNetwork(input_shape, n_output)
    model.summary()

    trainNetwork(model, training_dataset, validation_dataset)

    
#%%
    