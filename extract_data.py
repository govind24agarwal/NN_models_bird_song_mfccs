import json
import os
import math
from sys import path
import librosa
import warnings

DATASET_PATH = "/home/govind/Documents/ML/birdsong_data/train_audio"
JSON_PATH = "/home/govind/Documents/ML/NN_models_bird_song_mfccs/data"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
warnings.filterwarnings('ignore')


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Generating mfcc of every segment of every music clip of ever
        grnre and storing it into json file along with it's labels.

    Args:
        dataset_path (String): path of dataset of music clips
        json_path (String)): path of json file  where mfccs should be saved
        num_mfcc (int, optional): Number of mfcc coefficients to consider. Defaults to 13.
        n_fft (int, optional): Number of samples per frame. Defaults to 2048.
        hop_length (int, optional): Hop length for mfcc calculation. Defaults to 512.
        num_segments (int, optional): Number of segments in each music clip. Defaults to 5.
    """
    data = {
        "distinct_genres": [],
        "labels": [],
        "mfcc": []
    }
    failed = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Going through folder of each genre
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["distinct_genres"].append(semantic_label)
            print("Processing for genre: {}".format(semantic_label))
            data = {
                "distinct_genres": [],
                "labels": [],
                "mfcc": []
            }
            # Going through every music clip in it's genre's  folder
            for f in filenames:
                try:
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(
                        file_path, res_type="kaiser_fast", sr=None)

                    # mfccs
                    mfcc = librosa.feature.mfcc(
                        signal, sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # Saving mfccs
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    print("file: {}".format(f))
                except Exception as e:
                    print(e)
                    print("{} file failed".format(file_path))
                    failed += 1

            # Writing all data collected into json file
            name = semantic_label+".json"
            path = os.path.join(json_path, name)
            with open(path, "w") as fj:
                json.dump(data, fj, indent=4)

    print("{} files failed to load".format(failed))


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
