import numpy as np
import librosa
import csv
import scipy
import os


def get_features(directory, file):
    name = f"{directory}/{file}"
    y, sr = librosa.load(name, mono=True, duration=5)

    features = []
    features.append(file)
    features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])
    features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0])
    features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0])
    features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0])
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[0])
    features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[0])

    features.append(directory.split('/')[-1])
    return features

test_dir, _, test = next(os.walk("test"))
train_dir, _, train = next(os.walk("train"))
print(f"Test files: {len(test)}\nTrain files: {len(train)}")


buffer = []
buffer_size = 5000
buffer_counter = 0

header = ['filename']
header.extend([f'mfcc_mean{i}' for i in range(1, 21)])
header.extend([f'mfcc_std{i}' for i in range(1, 21)])
header.extend(['cent_mean', 'cent_std', 'cent_skew', 'rolloff_mean', 'rolloff_std', 'label'])

with open('test.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(header)
    for directory, files in [(test_dir, test)]:
        for file in files:
            features = get_features(directory, file)
            if buffer_counter + 1 == buffer_size:
                buffer.append(features)
                writer.writerows(buffer)
                print(f"- [{directory.split('/')[-1]}] Write {len(buffer)} rows")
                buffer = []
                buffer_counter = 0
            else:
                buffer.append(features)
                buffer_counter += 1
        if buffer:
            writer.writerows(buffer)
            print(f"- [{directory.split('/')[-1]}] Write {len(buffer)} rows")
        print(f"- [{directory.split('/')[-1]}] Writing complete")
        buffer = []
        buffer_counter = 0