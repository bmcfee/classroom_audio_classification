#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Compute time-aggregated statistics from stream-recorded features"""

import argparse
import numpy as np
import librosa
import StreamFeatures


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-s', '--short', dest='short_t',
                        type=float, default=5,
                        help='Duration (in seconds) of short-scale aggregation')
    parser.add_argument('-m', '--medium', dest='medium_t',
                        type=float, default=20,
                        help='Duration (in seconds) of mid-scale aggregation')
    parser.add_argument('-l', '--long', dest='long_t',
                        type=float, default=60,
                        help='Duration (in seconds) of long-scale aggregation')
    parser.add_argument('--hop', dest='hop_t',
                        type=float, default=2,
                        help='Duration (in seconds) of hop between frames')
    parser.add_argument(dest='input_file', type=str,
                        help='Input filename (npz)')
    parser.add_argument(dest='output_file', type=str,
                        help='Output filename (npz)')
    return parser.parse_args()


def summarize(data, short_t, medium_t, long_t, hop_t):

    out_data = dict()

    hop = librosa.time_to_frames(hop_t, sr=StreamFeatures.RATE,
                                 hop_length=StreamFeatures.HOP_LENGTH)

    min_frames = len(data['rms'])
    for (name, time) in [('short', short_t),
                         ('medium', medium_t),
                         ('long', long_t)]:
        frames = librosa.time_to_frames(time, sr=StreamFeatures.RATE,
                                        hop_length=StreamFeatures.HOP_LENGTH)

        for feature in ['rms', 'mfcc', 'contrast']:
            f_frame = librosa.util.frame(data[feature], axis=0,
                                         frame_length=frames,
                                         hop_length=hop)
            out_data[f"{name}_{feature}_mean"] = np.mean(f_frame, axis=1)
            out_data[f"{name}_{feature}_std"] = np.std(f_frame, axis=1)
            min_frames = min(min_frames, f_frame.shape[0])

    # Truncate everything to the same length prefix
    for key in out_data:
        out_data[key] = out_data[key][:min_frames]

    # Compute delta short-scale for medium and long aggregates
    for name in ['medium', 'long']:
        for feature in ['rms', 'mfcc', 'contrast']:
            for agg in ['mean', 'std']:
                out_data[f"{name}_{feature}_{agg}_delta"] = (
                    out_data[f"{name}_{feature}_{agg}"] - 
                    out_data[f"short_{feature}_{agg}"])

    # Get the global maximum RMS
    out_data['max_rms'] = np.max(data['rms']) * np.ones((min_frames, 1))

    return out_data


if __name__ == '__main__':
    args = parse_args()

    in_data = np.load(args.input_file)
    out_data = summarize(in_data, args.short_t, args.medium_t, args.long_t,
                         args.hop_t)
    np.savez(args.output_file, **out_data)
