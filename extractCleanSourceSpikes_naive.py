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

def frame_input_spike(x, y, new_resolution, neuron_spike_times, time):
    
    new_spike_id = np.multiply(x, new_resolution[0]) + y
    for elem in new_spike_id:
        neuron_spike_times[elem].append(time)
        
    return neuron_spike_times

def noise(frame_input):
    noise_mask = np.random.default_rng(seed=0).poisson(frame_input)
    frame_input += noise_mask
    return frame_input

def read_input_spikes_show_do_naive(filename, new_resolution):
    #
    Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polarity_add', filename)
    # lst = []
    
    neuron_spike_times_pos = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    neuron_spike_times_neg = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    
    scale = original_resolution / new_resolution
    
    noise_thr = np.sum([scale[0], scale[1]])
    # noise_thr = scale[0] * scale[1] / 4
    
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
            frame_input_image_neg = -frame_build(new_neg, new_resolution)
            
            # lst.append((time, np.max(frame_input_image)))
            
            # Noise filtering
            x_pos, y_pos = np.where(frame_input_image_pos[:,:] > noise_thr)
            x_neg, y_neg = np.where(frame_input_image_neg[:,:] < -noise_thr)
            
            neuron_spike_times_pos = frame_input_spike(x_pos, y_pos, new_resolution, neuron_spike_times_pos, time)
            neuron_spike_times_neg = frame_input_spike(x_neg, y_neg, new_resolution, neuron_spike_times_neg, time)
                
    return time, neuron_spike_times_pos, neuron_spike_times_neg