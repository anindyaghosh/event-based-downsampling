import argparse
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
from extractCleanSourceSpikes_do_combined_naive import read_input_spikes_show_do_naive
from extractCleanSourceSpikes_do_v1 import read_input_spikes_show_do_v1
from extractCleanSourceSpikes_do_v2 import read_input_spikes_show_do_v2
from method_analysis import build_full_polarity
# from noise_calc import noise_threshold

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from extractCleanSourceSpikes_global import read_input_spikes_show_global
from modified_cmp import mod_cmp

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--spikefile', type=str, required=True)
parser.add_argument('-r', '--resolution', type=str, required=True)
parser.add_argument('-p', '--polarity', type=str, required=True)
parser.add_argument('-v', '--version', type=str, required=True)
args = parser.parse_args()

# How many 
timesteps_per_frame = 25
viz_timestep = 1 # 33
original_resolution = np.array([640, 360]) # 640, 480
# new_resolution = (20, 20)
# filename = 'do-f_f_0.spikes'

def show_spikes_do(filename, new_resolution, polarity, version):
    
    if version not in 'global':
        if 'downsampled' in filename:
            sys.exit('Invalid combination of filename and version')
    
    # Convert string resolution tuple to np array
    if type(new_resolution) == str:
        if ',' in new_resolution:
            x_res, y_res = new_resolution.split(',')
            new_resolution = np.array([x_res, y_res], dtype=int)
        else:
            resolution = int(new_resolution)
            new_resolution = np.array([resolution, resolution])
    else:
        pass
    
    # Select downsampling method
    if version == 'naive':
        time, neuron_spike_times_pos, neuron_spike_times_neg = read_input_spikes_show_do_naive(filename, new_resolution)
    elif version == 'v1':
        time, neuron_spike_times_pos, neuron_spike_times_neg = read_input_spikes_show_do_v1(filename, new_resolution)
    elif version == 'v2':
        time, neuron_spike_times_pos, neuron_spike_times_neg = read_input_spikes_show_do_v2(filename, new_resolution)
    elif version == 'global':
        time, neuron_spike_times_pos, neuron_spike_times_neg = read_input_spikes_show_global(filename, new_resolution)
    else:
        sys.exit('Invalid version')
    
    time_view = [[] for x in np.arange(time + 1)]
    
    if version in ('v1', 'v2', 'global', 'naive'):
        if polarity == '+':
            neuron_spike_list = neuron_spike_times_pos
        elif polarity == '-':
            neuron_spike_list = neuron_spike_times_neg
        elif polarity == 'both': # cannot work for naive, only for visualisation purposes
            neuron_spike_list = []
    else:
        pass
        # neuron_spike_list = neuron_spike_times
    
    if 'both' not in polarity:
        for i, line in enumerate(neuron_spike_list):
            if line:
                [time_view[elem].append(i) for elem in line]
            else:
                pass
            
        frames = []
        
        for t, t_line in enumerate(time_view):
            frame = t // timesteps_per_frame
            
            t_line = np.asarray(t_line, dtype=int)
            x = np.floor_divide(t_line, new_resolution[0])
            y = np.remainder(t_line, new_resolution[0])
            
            frame_spike = np.histogram2d(x, y, [np.arange(new_resolution[1] + 1), 
                                                np.arange(new_resolution[0] + 1)])[0]
            frames.append((frame, frame_spike))
            
        num_frames = frames[-1][0] + 1
        timestep_inputs = np.zeros((new_resolution[1], new_resolution[0], num_frames), dtype=float)
        
        for i, f in frames:
            timestep_inputs[:,:,i] += f
            
    else:
        timestep_inputs = None
            
    res_string = '_'.join([str(num) for num in new_resolution])
    filehead, _ = filename.split('.')
    show_spikes_details = '_'.join([filehead, res_string, polarity, version])
    
    return timestep_inputs, show_spikes_details, polarity

# timestep_inputs, show_spikes_details, polarity = show_spikes_do('events_nature_downsample.spikes', "64, 36", 'both', 'naive')
# with open(show_spikes_details + '.npy', 'wb') as f:
#     np.save(f, timestep_inputs)

if len(sys.argv) > 4:
    timestep_inputs, show_spikes_details, polarity = show_spikes_do(args.spikefile, args.resolution, 
                                                                    args.polarity, args.version)
else:
    timestep_inputs, show_spikes_details, polarity = show_spikes_do(args.spikefile, original_resolution, 
                                                                    args.polarity, args.version)

if polarity == 'both':
    timestep_inputs = build_full_polarity(show_spikes_details, timesteps_per_frame)
    # show_spikes_details, dummy version -> events_nature_downsample_64_36_both_naive
else:
    with open(show_spikes_details + '.npy', 'wb') as f:
        np.save(f, timestep_inputs)

num_frames = timestep_inputs.shape[2] // viz_timestep

if not timestep_inputs.shape[2] % viz_timestep == 0:
    num_frames += 1
    
viz_timestep_inputs = np.zeros((*timestep_inputs.shape[:-1], num_frames))

for i in np.arange(num_frames):
    idx = [i*viz_timestep, (i+1)*viz_timestep]
    viz_timestep_inputs[:,:,i] += np.sum(timestep_inputs[:,:,idx[0]:idx[1]], axis=2)

fig, axis = plt.subplots()

if polarity == 'both':
    remainder_frames = timestep_inputs.shape[2] % viz_timestep
    
    if remainder_frames == 0:
        viz_timestep_inputs -= viz_timestep * 0.5
    else:
        viz_timestep_inputs[:,:,:-1] -= viz_timestep * 0.5
        viz_timestep_inputs[:,:,-1] -= remainder_frames * 0.5
else:
    pass

print(np.max(viz_timestep_inputs), np.min(viz_timestep_inputs))

midpoint = -np.min(viz_timestep_inputs) / (np.max(viz_timestep_inputs) - np.min(viz_timestep_inputs))
ratio = -np.min(viz_timestep_inputs) / np.max(viz_timestep_inputs)

print(midpoint, ratio)

newcmp = mod_cmp('bwr', polarity, midpoint, ratio)

input_image_data = viz_timestep_inputs[:,:,0]
input_image = axis.imshow(input_image_data, interpolation="nearest", cmap=newcmp, vmin=np.min(viz_timestep_inputs),
                          vmax=np.max(viz_timestep_inputs)) # float(viz_timestep))

divider = make_axes_locatable(axis)
cax = divider.append_axes("right", size="3%", pad=0.15)

fig.colorbar(input_image, cax=cax)

class Index:
    def __init__(self):
        self.ind = 0

    def _next(self, event):
        self.ind += 1
        i = self.ind % viz_timestep_inputs.shape[2]
        input_image_data = viz_timestep_inputs[:,:,i]
        input_image.set_array(input_image_data)
        time.set_text(str("Time = %u ms" %(i * timesteps_per_frame * viz_timestep)))
        plt.draw()

    def _prev(self, event):
        self.ind -= 1
        i = self.ind % viz_timestep_inputs.shape[2]
        input_image_data = viz_timestep_inputs[:,:,i]
        input_image.set_array(input_image_data)
        time.set_text(str("Time = %u ms" %(i * timesteps_per_frame * viz_timestep)))
        plt.draw()

callback = Index()
axprev = plt.axes([0.1, 0.9, 0.1, 0.07])
axnext = plt.axes([0.21, 0.9, 0.1, 0.07])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback._next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback._prev)

axtext = fig.add_axes([0.45, 0.8, 0.4, 0.05]) # plt.axes([left, bottom, width, height])
axtext.axis("off")
time = axtext.text(0.5, 0.5, str("Time = 0 ms"), ha="left", va="top")
# time = axis.text(0.75, 1.05, str("Time = 0 ms"), transform=axis.transAxes)
# Doesn't work with blit=True'

def updatefig(frame):
    global input_image_data, input_image, viz_timestep
    
    # Decay image data
    # input_image_data *= 0.9

    # Loop through all timesteps that occur within frame
    input_image_data = viz_timestep_inputs[:,:,frame]

    # Set image data
    input_image.set_array(input_image_data)
    
    time.set_text(str("Time = %u ms" %(frame*timesteps_per_frame*viz_timestep)))

    # Return list of artists which we have updated
    # **YUCK** order of these dictates sort order
    # **YUCK** score_text must be returned whether it has
    # been updated or not to prevent overdraw
    return input_image, time

# Play animation
# ani = animation.FuncAnimation(fig, updatefig, range(viz_timestep_inputs.shape[2]), interval=viz_timestep, blit=True, repeat=True)
plt.show()

writergif = animation.PillowWriter(fps=60)
writermp4 = animation.FFMpegWriter(fps=20)
# savefile = 'test_full_res.mp4'
savefile = '_'.join(['animation', show_spikes_details, '.mp4'])
# ani.save(savefile, writer = writermp4)