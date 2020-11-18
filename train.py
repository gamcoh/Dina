import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from resnet3d import Resnet3DBuilder

targets = pd.read_csv('data_csv/jester-v1-train.csv', header=None, sep = ";", index_col=0)
targets = targets.to_dict()[1]

targets_validation = pd.read_csv('data_csv/jester-v1-validation.csv', header=None, sep = ";", index_col=0)
targets_validation = targets_validation.to_dict()[1]

target_names = pd.read_csv('./data_csv/jester-v1-labels.csv', header=None)
target_names = target_names.to_dict()[0]

# Get the data directories
training_path = "./datasets/train/"
validation_path = "./datasets/val/"

train_dirs = os.listdir(training_path)
val_dirs = os.listdir(validation_path)

# The videos do not have the same number of frames, here we try to unify.
hm_frames = 30 # number of frames
# unify number of frames for each training
def get_unify_frames(path):
    # pick frames
    frames = os.listdir(path)
    frames_count = len(frames)
    # unify number of frames 
    if hm_frames > frames_count:
        # duplicate last frame if video is shorter than necessary
        frames += [frames[-1]] * (hm_frames - frames_count)
    elif hm_frames < frames_count:
        # If there are more frames, then sample starting offset
        frames = frames[0:hm_frames]
    return frames  


# ## Preprocess the data
def generator(dataset, batch_size=64):
    path = './datasets/'+dataset+'/'
    dirs = os.listdir(path)
    
    while True:
        batch_dirs = np.random.choice(dirs, size=batch_size)

        batch_x = []
        batch_y = []

        for d in batch_dirs:
            unify_frames = get_unify_frames(path+d)
            frames = [cv2.imread(path+d+'/'+f) for f in unify_frames]
            frames = [cv2.resize(frame, (64, 96)) for frame in frames]
            frames = np.array(frames)

            if dataset == 'train':
                label = targets[int(d)]
            elif dataset == 'val':
                label = targets_validation[int(d)]
            labels = np.zeros(len(target_names))
            index = list(target_names.values()).index(label)
            labels[index] = 1

            batch_x.append(frames)
            batch_y.append(labels)
        
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        batch_x = batch_x.reshape(-1, 30, 64, 96, 3)
        
        yield batch_x, batch_y

train_generator = generator('train', batch_size=64)
validation_generator = generator('val', batch_size=64)

model = Resnet3DBuilder.build_resnet_101((30, 64, 96, 3), 27)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])
callbacks = [
    ModelCheckpoint('./resnet3d_50_dropout_3c_64x96_b32.h5', monitor='val_accuracy', verbose=1, save_best_only=True),
    EarlyStopping(patience=5)
]
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=119_000//64,
                    validation_steps=15_000//64,
                    callbacks=callbacks,
                    epochs=30)
