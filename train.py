import os
import argparse
import tensorflow as tf

from utils.data_processing import (
    audio_preprocessing_tf, 
    one_hot_encoding, 
    read_dataset,
    input_dim
)
from utils.networks import create_model
from utils.visualization import visualize_training

RANDOM_SEED = 59
BATCH_SIZE = 128
TRAIN_SIZE = 0.7
EPOCHS = 10
AUDIO_DIR = './data/wav'
LABEL_PATH = './data/warblrb10k_public_metadata.csv'
SAVE_DIR = './checkpoint'
tf.random.set_seed(RANDOM_SEED) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=None
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None
    )
    parser.add_argument(
        '--is_visualize_training',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--is_save_model',
        type=bool,
        default=False
    )
    args = parser.parse_args()

    if args.data_dir != None:
        AUDIO_DIR = os.path.join(args.data_dir, 'wav')
        LABEL_PATH = os.path.hoin(args.data_dir, 'warblrb10k_public_metadata.csv')
    if args.n_epochs != None:
        EPOCHS = args.n_epochs
    if args.batch_size != None:
        BATCH_SIZE = args.batch_size

    audio_paths, labels = read_dataset(
        audio_dir=AUDIO_DIR,
        label_filepath=LABEL_PATH
    )

    train_n_samples = int(TRAIN_SIZE * len(audio_paths))

    train_audio_paths = audio_paths[:train_n_samples]
    train_labels = labels[:train_n_samples]

    val_audio_paths = audio_paths[train_n_samples:]
    val_labels = labels[train_n_samples:]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_audio_paths, train_labels)
    )
    train_ds = train_ds.map(
        lambda x, y: (audio_preprocessing_tf(x), one_hot_encoding(y)), 
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(train_n_samples//2).cache().prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_audio_paths, val_labels)
    )
    val_ds = val_ds.map(
        lambda x, y: (audio_preprocessing_tf(x), one_hot_encoding(y)), 
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE).batch(BATCH_SIZE)

    model = create_model(input_dim)
    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

    if args.is_visualize_training:
        visualize_training(history)

    if args.is_save_model:
        os.makedirs(SAVE_DIR, exist_ok=True)
        MODEL_NAME_DIR = f'model_{len(os.listdir(SAVE_DIR))}'
        os.makedirs(MODEL_NAME_DIR, exist_ok=True)
        model.save(MODEL_NAME_DIR)

if __name__ == '__main__':
    main()