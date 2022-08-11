import numpy as np
import os
import pandas as pd
from scipy import sparse

original_resolution = np.array([640, 360])
# original_resolution = (20, 20)
# new_resolution = 20
# filename = 'do-b_f_0.spikes'
timesteps_per_frame = 33

def frame_build(new_split, new_resolution):
    
    if new_split.shape[1] > 0:
        frame_input_image = np.histogram2d(np.abs(new_split[0,:]), np.abs(new_split[1,:]),
                                            [np.arange(new_resolution[1] + 1), np.arange(new_resolution[0] + 1)])[0]
    else:
        frame_input_image = np.zeros((new_resolution[1], new_resolution[0]))
    return frame_input_image

def frame_input_spike(x, y, new_resolution):
    
    spk_mat = np.zeros((new_resolution[0] * new_resolution[1], 1))
    
    new_spike_id = np.multiply(x, new_resolution[0]) + y
    spk_mat[new_spike_id] += 1
        
    return spk_mat

def timestep_spikes(frames, new_resolution):
    
    num_frames = frames[-1][0] + 1
    timestep_inputs = np.zeros((new_resolution[0] * new_resolution[1], 1, num_frames), dtype=float)
    timestep_adj = np.zeros((new_resolution[0] * new_resolution[1], 1, num_frames-1), dtype=float)
    for i, f in frames:
        timestep_inputs[:,:,i] += f
        
    for a in np.arange(num_frames-1):
        timestep_adj[:,:,a] += np.subtract(timestep_inputs[:,:,a+1], timestep_inputs[:,:,a])
        
    timestep_adj[timestep_adj[:,:] < 0] = 0
        
    return timestep_adj

def input_spikes(timestep_inputs, new_resolution):
    
    neuron_spike_times = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    
    for i in np.arange(timestep_inputs.shape[2]):
        ind = np.where(timestep_inputs[:,:,i] > 0)[0]
        for elem in ind:
            neuron_spike_times[elem].append(i)
        
    return neuron_spike_times

def read_input_spikes_show_do_v1(filename, new_resolution):
    #
    Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polarity_add33', filename)
    # lst = []
    
    scale = original_resolution / new_resolution
    # noise_thr = noise_threshold(new_resolution) + 5
    # noise_thr = int(np.rint(np.sqrt(np.multiply(original_resolution[0], original_resolution[1]) / 
    #                                 (new_resolution ** 2))))
    reshape_scale = np.sqrt(scale[0] * scale[1])
    noise_thr = np.ceil(reshape_scale / 4)
    
    frames_pos = []
    frames_neg = []
    
    with open(Path, "r") as input_spike_file:
        # Read input spike file
        for line in input_spike_file:
            
            # Split lines into time and keys
            time_string, keys_string = line.split(";")
            time = int(time_string)
            frame = time // timesteps_per_frame
            
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
            
            # Noise filtering
            x_pos, y_pos = np.where(frame_input_image[:,:] > noise_thr)
            x_neg, y_neg = np.where(frame_input_image[:,:] < -noise_thr)
            
            pos_add = frame_input_spike(x_pos, y_pos, new_resolution)
            neg_add = frame_input_spike(x_neg, y_neg, new_resolution)
            
            frames_pos.append((frame, pos_add))
            frames_neg.append((frame, neg_add))
            
        timestep_pos = timestep_spikes(frames_pos, new_resolution)
        timestep_neg = timestep_spikes(frames_neg, new_resolution)
        
        neuron_spike_times_pos = input_spikes(timestep_pos, new_resolution)
        neuron_spike_times_neg = input_spikes(timestep_neg, new_resolution)
        
        sim_time = int(time / timesteps_per_frame)
                
        # df = pd.DataFrame({'Time' : [i[0] for i in lst], 'Max': [i[1] for i in lst]})
        # df.to_csv('Max histogram bins per timestep_do-b_f.csv', index=False)
        
    return sim_time, neuron_spike_times_pos, neuron_spike_times_neg

# time, neuron_spike_times_pos, neuron_spike_times_neg = read_input_spikes_show_do(filename, new_resolution)