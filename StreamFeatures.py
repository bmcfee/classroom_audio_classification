#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Stream feature extraction for classroom assistant.

Note that this module only computes the frame-level features.
It does not compute the multi-scale time statistics that are used
in the model.

Instructions:
    1. Get a list of available devices:

        python StreamFeatures.py -l

        and select the device ID that you'll want to read from

    2. Run the feature extractor:

        python StreamFeatures.py -o features.npz -d {DEVICE_ID}

    3. Use CTRL-C to terminate reading

    4. Compute time-aggregated statistics from the features:

        python SummarizeFeatures.py features.npz features_summarized.npz

    The summarized output features will be collected in a npz dictionary.

    If you want to load them into the previous model, you can concatenate them
    as follows:

    >>> data = np.loadz('features_summarizes.npz')
    >>> X = np.hstack([data[f] for f in [
    ...                "short_rms_mean", "short_rms_std",
    ...                "short_mfcc_mean", "short_mfcc_std",
    ...                "short_contrast_mean", "short_contrast_std",
    ...                "medium_rms_mean_delta", "medium_rms_std_delta",
    ...                "medium_mfcc_mean_delta", "medium_mfcc_std_delta",
    ...                "medium_contrast_mean_delta", "medium_contrast_std_delta",
    ...                "long_rms_mean_delta", "long_rms_std_delta",
    ...                "long_mfcc_mean_delta", "long_mfcc_std_delta",
    ...                "long_contrast_mean_delta", "long_contrast_std_delta",
    ...                "max_rms",
    ...                ]
    ...              ])

"""

import argparse
import pyaudio
import librosa
from tqdm import tqdm
import sparklines
import numpy as np


RATE = 44100
FRAME_LENGTH = 2048
HOP_LENGTH = FRAME_LENGTH


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-o', '--output', dest='outfile',
                        type=str, required=False, help='Output file path')

    parser.add_argument('-d', '--device', dest='device_index',
                        type=int, required=False, help='Input device index')

    parser.add_argument('-l', '--list-devices', dest='list_devices',
                        action='store_true',
                        help='List devices')

    parser.add_argument('--frames-per-buffer', dest='frames',
                        default=22, type=int,  # default is ~1sec
                        help='Frames to process for each buffer')

    return parser.parse_args()


def list_devices():
    p = pyaudio.PyAudio()
    print("Input devices by ID")
    print("-------------------")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Device {i:2d} -- {info['name']}")


def analyze_stream(device_index, outfile, frames_per_buffer):
    p = pyaudio.PyAudio()

    data = dict(rms=[], mfcc=[], contrast=[])

    print(f"Opening device {device_index}")
    fd = p.open(input=True, output=False,
                input_device_index=device_index,
                format=pyaudio.paFloat32,
                channels=1,
                rate=RATE)

    print("Recording features.  Hit CTRL-C to terminate.")
    try:

        # Process frames with no overlap
        with tqdm() as pbar:
            while True:
                y = np.frombuffer(
                        fd.read(FRAME_LENGTH * frames_per_buffer,
                                exception_on_overflow=False),
                        dtype=np.float32)
                rms = librosa.feature.rms(y=y,
                                        frame_length=FRAME_LENGTH,
                                        hop_length=HOP_LENGTH,
                                        center=False)

                # Compute RMS in dB with an absolute reference point
                # NOTE: this differs from the original implementation,
                # which incorrectly computed RMS from the dB-scaled stft
                # magnitudes
                rms = librosa.amplitude_to_db(rms, ref=1)

                mfcc = librosa.feature.mfcc(y=y,
                                            sr=RATE,
                                            n_fft=FRAME_LENGTH,
                                            hop_length=HOP_LENGTH,
                                            center=False)
                contrast = librosa.feature.spectral_contrast(
                        y=y,
                        sr=RATE,
                        n_fft=FRAME_LENGTH,
                        hop_length=HOP_LENGTH,
                        center=False,
                        n_bands=6,
                        fmin=60)

                # output will be frames × features
                data['rms'].extend(rms.T)
                data['mfcc'].extend(mfcc.T)
                data['contrast'].extend(contrast.T)

                pbar.set_postfix_str(
                        sparklines.sparklines(np.maximum(rms[0] + 80, 0),
                                              minimum=0,
                                              maximum=96)[0],
                        refresh=False,
                )
                pbar.update()
                # TODO: put a sleep interval here to skip over some analysis


    except KeyboardInterrupt:
        print("End feature recording")
    finally:
        fd.close()
        p.terminate()
        print(f"Writing to {outfile}")
        data['rms'] = np.asarray(data['rms'])
        data['mfcc'] = np.asarray(data['mfcc'])
        data['contrast'] = np.asarray(data['contrast'])

        np.savez(outfile, **data)
        print("Done!")



if __name__ == "__main__":
    args = parse_args()

    if args.list_devices:
        list_devices()
    elif args.outfile is None:
        raise Exception(f"outfile must be provided")
    else:
        analyze_stream(device_index=args.device_index,
                       outfile=args.outfile,
                       frames_per_buffer=args.frames)
