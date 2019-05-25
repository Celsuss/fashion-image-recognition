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
import logging
import shutil
import math
import time
import sys
import os

print(tf.__version__)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
AUTOTUNE = tf.data.experimental.AUTOTUNE

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed) 
# tf.set_random_seed(seed)

#%%
DRAW = False

###################
# Hyperparameters #
###################
learning_rate = 0.001

height, width, channels = 28, 28, 1
n_epochs = 30
batch_size = 64
batch_count = 0

save_iterations = 1
checkpoint_path = './tmp'
tensorboard_base_path = './tf_logs'
tensorboard_traning_path = tensorboard_base_path + '/traning'
tensorboard_test_path = tensorboard_base_path + '/test'

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

def reshapeAndMakeOneHot(data, labels):
    data = reshapeTrainingData(data)
    labels = makeOneHot(labels)
    return data, labels

def preprocessData(image_raw):
    image = tf.cast(image_raw, tf.float32)
    image = image / 255.0
    return image

def createDataset(X, y, use_batch=True):
    global batch_count
    size = len(X)

    X_dataset = tf.data.Dataset.from_tensor_slices(X)
    y_dataset = tf.data.Dataset.from_tensor_slices(y)

    X_dataset = X_dataset.map(preprocessData, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((X_dataset, y_dataset))
    if use_batch:
        dataset = dataset.shuffle(buffer_size=size).batch(batch_size)
    else:
        dataset = dataset.shuffle(buffer_size=size).batch(size)

    if batch_count == 0:
        batch_count = math.ceil(size/batch_size)
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
        print('Restored checkpoint from {}'.format(ckptManager.latest_checkpoint))
    else:
        print('No checkpoints found, start training at epoch 0')

def saveCheckpoint(ckptManager):
    save_path = ckptManager.save()
    print("Saved a checkpoint at {}\n".format(save_path))

#%%
##############################
# Loss functions and metrics #
##############################
def getLossOp():
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True) 

def getLossAndAccuracyMetrics(name):
    train_loss = tf.keras.metrics.Mean(name=name+'_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name=name+'_accuracy')
    return train_loss, train_accuracy

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

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_op = getLossOp()
    train_loss, train_accuracy = getLossAndAccuracyMetrics('train')
    validation_loss, validation_accuracy = getLossAndAccuracyMetrics('validation')

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
                epoch+1, n_batch, batch_count, train_loss.result(), train_accuracy.result()), end='\r')

            # TODO: Move this to function
            tf.summary.scalar('training loss', train_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('training accuracy', train_accuracy.result(), step=optimizer.iterations)
            train_loss.reset_states()
            train_accuracy.reset_states()

        for validation_image, validation_label in validation_data:
            testStep(validation_image, validation_label, model, loss_op, validation_loss, validation_accuracy)

        print('\nValidation loss {}, Validation accuracy {} \nTime for epoch {} is {} sec'.format(validation_loss.result(), validation_accuracy.result(), epoch+1, time.time()-startTime))
        # TODO: Move this to function
        tf.summary.scalar('validation loss', validation_loss.result(), step=optimizer.iterations)
        tf.summary.scalar('validation accuracy', validation_accuracy.result(), step=optimizer.iterations)
        validation_loss.reset_states()
        validation_accuracy.reset_states()

        if (epoch+1) % save_iterations == 0:
            saveCheckpoint(ckptManager)

    return model
                

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
def testStep(images, labels, model, loss_op, test_loss, test_accuracy):
    predictions = model(images)
    loss = loss_op(predictions, labels)
    test_loss(loss)
    test_accuracy(labels, predictions)

#%%
####################
# Test the network #
####################
def testNetwork(model, test_data):
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_op = getLossOp()
    test_loss, test_accuracy = getLossAndAccuracyMetrics('test')

    for test_image, test_label in test_data:
        testStep(test_image, test_label, model, loss_op, test_loss, test_accuracy)
    print('Test loss {}, Test accuracy {}'.format(test_loss.result(), test_accuracy.result()))

#%%
def deleteCheckpoints():
    answer = ''

    while answer != 'y' and answer != 'Y' and answer != 'N' and answer != 'n':
        print('Are you sure you want to remove all checkpoints? (Y/N)')
        answer = input()
        
    if answer == 'Y' or answer == 'y':
        paths = [checkpoint_path, tensorboard_traning_path, tensorboard_test_path]
        for path in paths:
            if os.path.isdir(path):
                shutil.rmtree(path)

def handleArguments():
    global DRAW
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '-d':
                DRAW = True
            elif arg == '-c':
                deleteCheckpoints()
                

#%%
if __name__ == '__main__':
    # if len(sys.argv) == 2:
    #     if sys.argv[1] == '-d':
    #         DRAW = True
    handleArguments()

    print("** Fashion Recognizer **")
    X_train, y_train, X_val, y_val, X_test, y_test = loadData('Data/')
    X_train, y_train = reshapeAndMakeOneHot(X_train, y_train)
    X_val, y_val = reshapeAndMakeOneHot(X_val, y_val)
    X_test, y_test = reshapeAndMakeOneHot(X_test, y_test)

    drawImage(X_train[10])

    input_shape = X_train.shape[1:]     # (28, 28, 1)
    n_output = y_train.shape[-1]        # 10

    training_dataset = createDataset(X_train, y_train)
    validation_dataset = createDataset(X_val, y_val, use_batch=False)
    test_dataset = createDataset(X_test, y_test, use_batch=False)

    model = createNetwork(input_shape, n_output)
    model.summary()

    # Create summary logging
    train_summary_writer = tf.summary.create_file_writer(tensorboard_traning_path)
    test_summary_writer = tf.summary.create_file_writer(tensorboard_test_path)

    with train_summary_writer.as_default():
        model = trainNetwork(model, training_dataset, validation_dataset)
    with test_summary_writer.as_default():
        testNetwork(model, test_dataset)

    
#%%
    