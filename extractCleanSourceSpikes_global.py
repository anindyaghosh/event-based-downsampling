import numpy as np
import os

# How many 
# new_resolution = original_resolution
# filename = 'do-f_f_0.spikes'

def frame_build(new_split, new_resolution):
    
    if new_split.shape[1] > 0:
        frame_input_image = np.histogram2d(np.abs(new_split[0,:]), np.abs(new_split[1,:]),
                                            [np.arange(new_resolution[1] + 1), np.arange(new_resolution[0] + 1)])[0]
    else:
        frame_input_image = np.zeros((new_resolution[1], new_resolution[0]))
    return frame_input_image

def read_input_spikes_show_global(filename, new_resolution):
    #
    Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'do_dataset\polarity_add', filename)
    # lst = []
    
    if 'downsampled' in filename:
        original_resolution = np.array([64, 36])
    else:
        original_resolution = np.array([640, 360])
    
    neuron_spike_times_pos = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    neuron_spike_times_neg = [[] for x in np.arange(new_resolution[0] * new_resolution[1])]
    
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
                
            frame_spike = np.vstack((frame_input_x, frame_input_y))
            new_neg = np.take(frame_spike, neg_track, axis=1)
            
            new_neg[0, :] *= -1
                
            pos_track = np.setdiff1d(np.arange(frame_spike.shape[1]), neg_track)
            new_pos = np.take(frame_spike, pos_track, axis=1)
                
            # Take histogram so as to assemble frame
            frame_input_image_pos = frame_build(new_pos, new_resolution)
            frame_input_image_neg = frame_build(new_neg, new_resolution)
            
            # Eliminate temporal downsampling jitter
            frame_input_image = np.subtract(frame_input_image_pos, frame_input_image_neg)
            
            x_pos, y_pos = np.where(frame_input_image[:,:] > 0)
            x_neg, y_neg = np.where(frame_input_image[:,:] < 0)
            
            pos_spks = np.ravel_multi_index([x_pos, y_pos], (new_resolution[1], new_resolution[0]))
            neg_spks = np.ravel_multi_index([x_neg, y_neg], (new_resolution[1], new_resolution[0]))
            
            [neuron_spike_times_pos[elem].append(time) for elem in pos_spks]
            [neuron_spike_times_neg[elem].append(time) for elem in neg_spks]
            
            # df = pd.DataFrame({'Time' : [i[0] for i in lst], 'Max': [i[1] for i in lst]})
            # df.to_csv('Max histogram bins per timestep_b_f.csv', index=False)
            
        return time, neuron_spike_times_pos, neuron_spike_times_neg
    
# time, neuron_spike_times_pos, neuron_spike_times_neg = read_input_spikes_show_global(filename, new_resolution)