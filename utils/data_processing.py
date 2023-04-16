import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import os
import argparse
import matplotlib.pyplot as plt

frame_length = 256
frame_step = 160
fft_length = 384
input_dim = fft_length // 2 + 1
max_duration = 25
target_sr = 16000
n_mels = 128
n_mfcc = 40

def audio_preprocessing_librosa(
        audio_path, 
        type='spectrogram', 
        is_normalize=True,
        is_fix_duration=True
    ):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    if is_fix_duration:
        audio = librosa.util.fix_length(audio, 
            size=int(sr * max_duration)
        )
    audio = audio.astype(np.float32)
    stfts = librosa.stft(
        audio,
        n_fft=fft_length,
        hop_length=frame_step,
        win_length=frame_length,
    )

    stfts = np.abs(stfts)
    spectrogram = stfts ** 2

    if type == 'mel_spectrogram':
        spectrogram = librosa.feature.melspectrogram(
            S=spectrogram, 
            n_mels=n_mels, 
            sr=sr
        )
    elif type == 'mfccs':
        spectrogram = np.log(spectrogram)
        spectrogram = librosa.feature.mfcc(
            S=spectrogram,
            n_mfcc=n_mfcc,
            sr=sr
        )

    spectrogram = librosa.amplitude_to_db(stfts)

    if is_normalize:
        means = np.mean(spectrogram, axis=1, keepdims=True)
        stddevs = np.std(spectrogram, axis=1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    spectrogram = spectrogram.T

    return spectrogram

def audio_preprocessing_tf(audio_path, type='spectrogram'):
    audio_file = tf.io.read_file(audio_path)
    audio, sr = tf.audio.decode_wav(
        audio_file,
        desired_samples=int(max_duration * target_sr)
    )

    audio = tf.cast(audio, tf.float32) 
    audio = tf.squeeze(audio, axis=-1)

    stfts = tf.signal.stft(
        audio, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )

    spectrogram = tf.math.abs(stfts) 
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=80)

    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram

def one_hot_encoding(label, n_classes=2):
    vector_one_hot = tf.one_hot(label, n_classes)

    return vector_one_hot

def read_dataset(audio_dir, label_filepath):
    label_df = pd.read_csv(label_filepath)
    label_df['itemid'] = label_df['itemid'].apply(lambda x: os.path.join(audio_dir, x + '.wav'))
    audio_paths = label_df['itemid'].values
    labels = label_df['hasbird'].values
    
    return audio_paths, labels

def visualize(in_features, type='spectrogram'):
    plt.figure(figsize=(10, 6))
    plt.imshow(in_features.T, aspect='auto', origin='lower', cmap='jet')
    plt.xlabel('Time (cs)')
    plt.ylabel('Hz')
    plt.colorbar()
    plt.title(f'{type} visualization')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file'
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        choices=['spectrogram', 'mel_spectrogram', 'mfccs']
    )
    parser.add_argument(
        '--is_visualize',
        type=bool,
        default=False,
        help='Visualize the extracted features of input audio'
    )
    args = parser.parse_args()

    audio_features = audio_preprocessing_librosa(
        args.audio,
        args.features,
        is_normalize=False,
        is_fix_duration=False
    ) 

    if args.is_visualize:
        visualize(
            in_features=audio_features,
            type=args.features
        )

    print(f'Audio {args.features} shape: {audio_features.shape}')

if __name__ == '__main__':
    main()