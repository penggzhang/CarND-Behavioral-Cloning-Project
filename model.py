import pandas as pd
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import model_from_json

import os, json
from pathlib import Path


def load_log(log_fpath, header):
    """
    Load driving log into a data frame.
    log_fpath: Path of driving log csv file.
    header: None if there is no header in driving log csv, 
            0 if with one-line header, default 0.
    """
    driving_log_file = log_fpath
    log_df = pd.read_csv(driving_log_file, 
                         header=header,
                         names=['center','left','right','steer','throttle','break', 'speed'],
                         index_col = False)
    # Give data summary
    print("\nShape of data frame: {}".format(log_df.shape))
    return log_df


def load_all_logs(file_list):
    """
    Load multiple driving logs into one data frame.
    file_list: a list of file paths.
    """
    df_list = []
    for fpath in file_list:
        df = load_log(fpath, 0)
        df_list.append(df)
    log_df = pd.concat(df_list)
    print("\nMultiple logs loaded.")
    return log_df


def filter_samples(log_df, throttle_threshold):
    """
    Remove samples with throttle below threshold.
    """
    log_df = log_df[log_df['throttle'] >= throttle_threshold]
    # Reset the index of filtered data frame.
    log_df = log_df.reset_index(drop=True)

    nb_samples = log_df.shape[0]
    print("\nNumber of samples after filtering: {}".format(nb_samples))
    return log_df, nb_samples


def get_image_local_path(df, cameras, IMG_parent_dir):
    """
    Modifying path string in log to local path.
    """
    n = df.shape[0]
    for c in cameras:
        for i in range(n):
            source_path = df[c][i].strip()
            fname = source_path.split('/')[-1]
            local_path = "./" + IMG_parent_dir + '/IMG/' + fname
            df.set_value(i, c, local_path.strip())
    # Check modified path strings.
    print("\nFile path of 1st sample:")
    for c in cameras:
        print(df[c][0])    
    return df


def shuffle_samples(log_df):
    """
    Shuffle samples by log.
    """
    log_df = log_df.sample(frac=1).reset_index(drop=True)
    print("\nSamples shuffled.")
    return log_df


def split_train_and_valid(log_df, nb_samples, valid_split=0.2):
    """
    Split train and validation sets by log.
    """
    # Generate a Boolean mask for splitting.
    msk = (np.random.rand(nb_samples) < valid_split)

    # Split with the mask and reset index.
    valid_log = log_df[msk].reset_index(drop=True)
    train_log = log_df[~msk].reset_index(drop=True)

    nb_samples_train = train_log.shape[0]
    nb_samples_valid = valid_log.shape[0]
    assert nb_samples_train + nb_samples_valid == nb_samples
    print("\nNumbers of train and validation samples: {0}, {1}".\
          format(nb_samples_train, nb_samples_valid))

    return train_log, valid_log, nb_samples_train, nb_samples_valid


def load_image(fpath):
    """
    Load image, and then convert it to RGB.
    """
    image = cv2.imread(fpath)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(fpath):
    """
    1) Remove the hood from image. Crop top 50 pixels 
       and bottom 30 pixels.
    2) Downsize image by a factor of 2 on either dimension. 
    """
    image = load_image(fpath)
    row, col = image.shape[0], image.shape[1]
    
    # Crop top and bottom pixels.
    top_crop = 50
    btm_crop = 30
    image = image[top_crop:(row - btm_crop), :]
    
    # Resize to new row and col.
    downsize = 2
    new_row = (row - top_crop - btm_crop) // downsize
    new_col = col // downsize
    image = cv2.resize(image, 
                       (new_col, new_row), 
                       interpolation=cv2.INTER_AREA)
        
    return image


def augment_left_and_right_steers(log, steer_correction):
    """
    Incorporate left and right images. Add a steer correction
    for the left image, which means we have to steer further
    right back to center. Subtract the correction for right image,
    which means we have to steer further left back to center.
    """
    base = np.array(log['steer'])
    log = log.assign(left_steer=pd.Series(base+steer_correction).values)
    log = log.assign(right_steer=pd.Series(base-steer_correction).values)
    assert len(log['left_steer']) == len(log['right_steer']) == len(log['steer'])
    print("\nSteers were generated for left and right images.")
    return log


def generate_data(log, cameras, d, batch_size):
    """
    Generate data batch by batch. Each batch contains
    a tuple of batch images and steers.
    """
    n = log.shape[0]
    while 1: 
        for offset in range(0, n, batch_size):
            batch_images, batch_steers = [], []
            k = min(n - offset, batch_size)
            for i in range(k):
                row = offset + i
                for c in cameras:
                    fpath = log[c][row]
                    img = preprocess_image(fpath)
                    steer = log[d[c]][row]
                    
                    batch_images.append(img)
                    batch_steers.append(steer)
                    
                    # Mirror center image and label it with negative steer
                    if c == 'center':
                        batch_images.append(cv2.flip(img, 1))
                        batch_steers.append(-float(steer))
                    
            batch_images = np.array(batch_images)
            batch_steers = np.array(batch_steers)
            yield batch_images, batch_steers

            
def create_model(input_shape):
    """
    Design and create a CNN model.
    """
    model = Sequential()

    # Normalization
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))

    # 1x1 convolutional layer
    model.add(Convolution2D(3, 1, 1, border_mode='valid', name='conv0'))
    model.add(Activation('relu'))

    # 5 convolutional layers
    model.add(Convolution2D(24, 3, 3, border_mode='valid', name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3, border_mode='valid', name='conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='valid', name='conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 1, 1, border_mode='valid', name='conv5'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    # 3 fully connected layers
    model.add(Dense(100, W_regularizer=l2(5e-4), name='fc1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, W_regularizer=l2(5e-4), name='fc2'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, W_regularizer=l2(5e-4), name='fc3'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, name='output'))
    
    return model


def compile_model(model, lr):
    """
    Compile the model. Fine tune the learning rate, lr,
    along training sessions.
    """
    # Optimizer: ADAM
    adam = Adam(lr=lr)
    # Optimization objective: mean square error.
    model.compile(optimizer=adam, loss='mse')
    return model


def save_model(model, json_file, weights_file):
    """
    Save a model.
    """
    json_string = model.to_json()
    
    # Save architecture to JSON file.
    if Path(json_file).is_file():
        os.remove(json_file)
    with open(json_file, 'w') as f:
        json.dump(json_string, f)

    # Save weights to HDF5 file.
    if Path(weights_file).is_file():
        os.remove(weights_file)
    model.save_weights(weights_file)
    
    print("Model saved.")

    
def train(nb_epochs, save, model_id):
    global model, cameras, d_train, d_valid, train_log, valid_log
    global nb_samples_train, nb_samples_valid, batch_size
    
    nb_samples_per_epoch = (len(cameras) + 1) * nb_samples_train
    # Note: 3 cameras + 1 mirrored
    nb_val_samples = nb_samples_valid * 2
    # Note: Multiply 2 for the mirrored
    
    for i in range(nb_epochs):

        # Create generators for train and validation data.
        train_generator = generate_data(train_log, cameras, d_train, batch_size)
        valid_generator = generate_data(valid_log, ['center'], d_valid, batch_size)

        # Train model on data batch-by-batch.
        # Also validate on validation data batch-by-batch.
        history = model.fit_generator(generator=train_generator,
                                      samples_per_epoch=nb_samples_per_epoch,
                                      nb_epoch=1,
                                      validation_data=valid_generator,
                                      nb_val_samples=nb_val_samples)

        if save:
            # Save models for each training epoch.
            json_file = 'models/model_' + str(model_id) + '.json'
            weights_file = 'models/model_' + str(model_id) + '.h5'
            save_model(model, json_file, weights_file)
            model_id += 1
            # Note: Make local directory 'models' beforehand.
            
    return model


def load_model(json_file, weights_file):
    """
    Load a pre-trained model.
    """
    with open(json_file, 'r') as f:
        model = model_from_json(json.loads(f.read()))
    model.load_weights(weights_file, by_name=True)
    print("Model loaded.")
    return model  


######### Train pipeline #########

## Part 1: Load and prepare data.

# Load driving log    
log_fpath = "data/driving_log.csv"
header = 0
log_df = load_log(log_fpath, header)

# Load multiple driving logs
#file_list = []
#log_df = load_all_logs(file_list)

# Filter samples
throttle_threshold = 0.1
log_df, nb_samples = filter_samples(log_df, throttle_threshold)

# Modify image log to local path
cameras = ['center', 'left', 'right'] 
IMG_parent_dir = "data"
log_df = get_image_local_path(log_df, cameras, IMG_parent_dir)

# Shuffle samples
log_df = shuffle_samples(log_df)
    
# Split training and valiation sets by log
valid_split = 0.1
train_log, valid_log, nb_samples_train, nb_samples_valid = \
    split_train_and_valid(log_df, nb_samples, valid_split)
    
# Augment steers for left and right images for training set.
steer_correction = 0.06
train_log = augment_left_and_right_steers(train_log, steer_correction)


## Part 2.a: Create a model.
input_shape = (40, 160, 3)
model = create_model(input_shape)

## Part 2.b: Load a pre-trained model.
#json_file = "model.json"
#weights_file = "model.h5"
#model = load_model(json_file, weights_file)


## Part 3: Compile the model.
lr = 1e-3
model = compile_model(model, lr)


## Part 4: Setup parameters for Keras fit_generators.
d_train = {'left'  : 'left_steer', 
           'center': 'steer',
           'right' : 'right_steer'}
d_valid = {'center': 'steer'}
batch_size = 64


## Part 5: Train the model.
nb_epochs = 5
save = True
model_id = 1
model = train(nb_epochs, save, model_id)
















