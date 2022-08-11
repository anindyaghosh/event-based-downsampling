# Questions?
# 1. What happens if positive and negative spike at the same time?
# 2. 

import numpy as np
import os
import pandas as pd
from scipy import sparse

original_resolution = np.array([640, 360])
# original_resolution = (20, 20)
# new_resolution = 20
# filename = 'do-b_f_0.spikes'

def frame_build(new_split, new_resolution):
    
    if new_split.shape[1] > 0:
        frame_input_image = np.histogram2d(np.abs(new_split[0,:]), np.abs(new_split[1,:]),
                                            [np.arange(new_resolution[1] + 1), np.arange(new_resolution[0] + 1)])[0]
    else:
        frame_input_image = np.zeros((new_resolution[1], new_resolution[0]))
    return frame_input_image

def read_input_spikes_show_do_v2(filename, new_resolution):
    #
    Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polarity_add', filename)
    # lst = []
    
    neuron_spike_times_pos = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    neuron_spike_times_neg = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    
    scale = original_resolution / new_resolution
    # noise_thr = noise_threshold(new_resolution) + 5
    # noise_thr = int(np.rint(np.sqrt(np.multiply(original_resolution[0], original_resolution[1]) / 
    #                                 (new_resolution ** 2))))
    # reshape_scale = np.rint(np.sqrt(scale[0] * scale[1]))
    # noise_thr = int(np.rint(np.sqrt(reshape_scale * reshape_scale / 2)))
    noise_thr = scale[0] * scale[1] * (100 / 27.5) * 0.225 # 0.25
    
    # noise_thr = 5
    
    frame_spike = np.zeros((new_resolution[1], new_resolution[0]), dtype=float)
    
    with open(Path, "r") as input_spike_file:
        # Read input spike file
        for line in input_spike_file:
            
            # Split lines into time and keys
            time_string, keys_string = line.split(";")
            time = int(time_string)
            
            # Load spikes into numpy array
            frame_input_spikes = np.asarray(keys_string.split(","), dtype=str)
            neg_track = [i for i, elem in enumerate(frame_input_spikes) if '-' in elem]
            frame_input_spikes = np.abs(frame_input_spikes.astype(int))
    
            # Split into X and Y
            frame_input_x = np.floor_divide(frame_input_spikes, original_resolution[0])
            frame_input_y = np.remainder(frame_input_spikes, original_resolution[0])
            
            # X and Y now correspond to rows and columns respectively
            new_x = np.floor(frame_input_x / scale[1])
            new_y = np.floor(frame_input_y / scale[0])
            
            new_ = np.vstack((new_x, new_y))
            new_ = new_.astype(int)
            
            new_neg = np.take(new_, neg_track, axis=1)
            
            new_neg[0, :] *= -1
            
            pos_track = np.setdiff1d(np.arange(new_.shape[1]), neg_track)
            new_pos = np.take(new_, pos_track, axis=1)
            
            # Take histogram so as to assemble frame
            frame_input_image_pos = frame_build(new_pos, new_resolution)
            frame_input_image_neg = frame_build(new_neg, new_resolution)
            
            frame_input_image = np.subtract(frame_input_image_pos, frame_input_image_neg)
            
            # lst.append((time, np.max(frame_input_image)))
            
            # Graded addition of spikes
            frame_spike += frame_input_image
            
            for i in np.arange(new_resolution[1]):
                for j in np.arange(new_resolution[0]):
                    pix = i * new_resolution[0] + j
                    if frame_spike[i, j] > noise_thr:
                        neuron_spike_times_pos[pix].append(time)
                        frame_spike[i, j] = 0
                    elif frame_spike[i, j] < -noise_thr:
                        neuron_spike_times_neg[pix].append(time)
                        frame_spike[i, j] = 0
                    else:
                        pass
                
    return time, neuron_spike_times_pos, neuron_spike_times_neg