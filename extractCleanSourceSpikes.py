import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))

class extractSourceSpikes:
    def __init__(self, filename, new_resolution, version):
        self.filename = filename
        self.original_resolution = np.array([640, 480])
        self.new_resolution = new_resolution
        self.version = version
        self.timesteps_per_frame = 33
        
    def path(self):
        outDir = 'polarity_add33' if self.version == 'v1' else 'polarity_add'
        self.Path = os.path.join(root, 'do_dataset', outDir, self.filename)
    
    def synth_condition(self):
        if 'synthSpikes' in self.filename:
            self.original_resolution = np.array([20, 20])
    
    def resolution_check(self):
        self.synth_condition()
        if self.version == 'global':
            self.new_resolution = self.original_resolution
        self.scale = self.original_resolution / self.new_resolution
        
    def noise_threshold(self):
        if self.version == 'naive':
            self.noise_thr = 0
        elif self.version == 'v1':
            reshape_scale = np.sqrt(self.scale[0] * self.scale[1])
            self.noise_thr = np.ceil(reshape_scale / 4)
        elif self.version == 'v2':
            self.noise_thr = self.scale[0] * self.scale[1] * (100 / 27.5) * 0.225
    
    def frame_build(self, new_split):
        
        if new_split.shape[1] > 0:
            frame_input_image = np.histogram2d(np.abs(new_split[0,:]), np.abs(new_split[1,:]),
                                                [np.arange(self.new_resolution[1] + 1), np.arange(self.new_resolution[0] + 1)])[0]
        else:
            frame_input_image = np.zeros((self.new_resolution[1], self.new_resolution[0]))
        return frame_input_image
    
# ----------------------------------------------------------------------------
# V1  
    
    def frame_input_spike(self, x, y):
        
        spk_mat = np.zeros((self.new_resolution[0] * self.new_resolution[1], 1))
        
        new_spike_id = np.multiply(x, self.new_resolution[0]) + y
        spk_mat[new_spike_id] += 1
            
        return spk_mat

    def timestep_spikes(self, frames):
        
        num_frames = frames[-1][0] + 1
        timestep_inputs = np.zeros((self.new_resolution[0] * self.new_resolution[1], 1, num_frames), dtype=float)
        timestep_adj = np.zeros((self.new_resolution[0] * self.new_resolution[1], 1, num_frames-1), dtype=float)
        for i, f in frames:
            timestep_inputs[:,:,i] += f
            
        for a in np.arange(num_frames-1):
            timestep_adj[:,:,a] += np.subtract(timestep_inputs[:,:,a+1], timestep_inputs[:,:,a])
            
        timestep_adj[timestep_adj[:,:] < 0] = 0
            
        return timestep_adj

    def input_spikes(self, timestep_inputs):
        
        neuron_spike_times = [[] for x in np.arange(self.new_resolution[0] * self.new_resolution[1])]
        
        for i in np.arange(timestep_inputs.shape[2]):
            ind = np.where(timestep_inputs[:,:,i] > 0)[0]
            for elem in ind:
                neuron_spike_times[elem].append(i)
            
        return neuron_spike_times

# ----------------------------------------------------------------------------

    def naive(self):
        
        # Take histogram so as to assemble frame
        frame_input_image_pos = self.frame_build(self.new_pos)
        frame_input_image_neg = -self.frame_build(self.new_neg)
        
        # lst.append((time, np.max(frame_input_image)))
        
        # Noise filtering
        self.x_pos, self.y_pos = np.where(frame_input_image_pos[:,:] > self.noise_thr)
        self.x_neg, self.y_neg = np.where(frame_input_image_neg[:,:] < -self.noise_thr)
        
        new_spike_id_pos = np.multiply(self.x_pos, self.new_resolution[0]) + self.y_pos
        new_spike_id_neg = np.multiply(self.x_neg, self.new_resolution[0]) + self.y_neg
        
        for elem in new_spike_id_pos:
            self.neuron_spike_times_pos[elem].append(self.time)
            
        for elem in new_spike_id_neg:
            self.neuron_spike_times_neg[elem].append(self.time)
            
    def v1(self):
        
        # Take histogram so as to assemble frame
        frame_input_image_pos = self.frame_build(self.new_pos)
        frame_input_image_neg = self.frame_build(self.new_neg)
        
        frame_input_image = np.subtract(frame_input_image_pos, frame_input_image_neg)
        
        # lst.append((time, np.max(frame_input_image)))
        
        # Noise filtering
        self.x_pos, self.y_pos = np.where(frame_input_image[:,:] > self.noise_thr)
        self.x_neg, self.y_neg = np.where(frame_input_image[:,:] < -self.noise_thr)
        
        pos_add = self.frame_input_spike(self.x_pos, self.y_pos)
        neg_add = self.frame_input_spike(self.x_neg, self.y_neg)
        
        self.frames_pos.append((self.frame, pos_add))
        self.frames_neg.append((self.frame, neg_add))
        
    def v1_finish(self):
        timestep_pos = self.timestep_spikes(self.frames_pos)
        timestep_neg = self.timestep_spikes(self.frames_neg)
        
        neuron_spike_times_pos = self.input_spikes(timestep_pos)
        neuron_spike_times_neg = self.input_spikes(timestep_neg)
        
        sim_time = int(self.time / self.timesteps_per_frame)
        
        return sim_time, neuron_spike_times_pos, neuron_spike_times_neg
    
    def v2(self):
        frame_input_image_pos = self.frame_build(self.new_pos)
        frame_input_image_neg = self.frame_build(self.new_neg)
        
        frame_input_image = np.subtract(frame_input_image_pos, frame_input_image_neg)
        
        # lst.append((time, np.max(frame_input_image)))
        
        # Graded addition of spikes
        self.frame_spike += frame_input_image
        
        for i in np.arange(self.new_resolution[1]):
            for j in np.arange(self.new_resolution[0]):
                pix = i * self.new_resolution[0] + j
                if self.frame_spike[i, j] > self.noise_thr:
                    self.neuron_spike_times_pos[pix].append(self.time)
                    self.frame_spike[i, j] = 0
                elif self.frame_spike[i, j] < -self.noise_thr:
                    self.neuron_spike_times_neg[pix].append(self.time)
                    self.frame_spike[i, j] = 0
                else:
                    pass
                
    def spike_global(self):
        
        # Take histogram so as to assemble frame
        frame_input_image_pos = self.frame_build(self.new_pos)
        frame_input_image_neg = self.frame_build(self.new_neg)
        
        # Eliminate temporal downsampling jitter
        frame_input_image = np.subtract(frame_input_image_pos, frame_input_image_neg)
        
        self.x_pos, self.y_pos = np.where(frame_input_image[:,:] > 0)
        self.x_neg, self.y_neg = np.where(frame_input_image[:,:] < 0)
        
        pos_spks = np.ravel_multi_index([self.x_pos, self.y_pos], (self.new_resolution[1], self.new_resolution[0]))
        neg_spks = np.ravel_multi_index([self.x_neg, self.y_neg], (self.new_resolution[1], self.new_resolution[0]))
        
        [self.neuron_spike_times_pos[elem].append(self.time) for elem in pos_spks]
        [self.neuron_spike_times_neg[elem].append(self.time) for elem in neg_spks]
            
    def read_input_spikes(self):
        self.path()
        self.resolution_check()
        self.noise_threshold()
        
        self.neuron_spike_times_pos = [[] for x in np.arange(self.new_resolution[0] * self.new_resolution[1])]
        self.neuron_spike_times_neg = [[] for x in np.arange(self.new_resolution[0] * self.new_resolution[1])]
        
        self.frames_pos = []
        self.frames_neg = []
        
        self.frame_spike = np.zeros((self.new_resolution[1], self.new_resolution[0]), dtype=float)
        
        with open(self.Path, "r") as input_spike_file:
            # Read input spike file
            for line in input_spike_file:
                
                # Split lines into time and keys
                time_string, keys_string = line.split(";")
                self.time = int(time_string)
                self.frame = self.time // self.timesteps_per_frame
                
                # Load spikes into numpy array
                frame_input_spikes = np.asarray(keys_string.split(","), dtype=str)
                neg_track = [i for i, elem in enumerate(frame_input_spikes) if '-' in elem]
                frame_input_spikes = np.abs(frame_input_spikes.astype(int))
        
                # Split into X and Y
                frame_input_x = np.floor_divide(frame_input_spikes, self.original_resolution[0])
                frame_input_y = np.remainder(frame_input_spikes, self.original_resolution[0])
                
                if self.version == 'global':
                    new_ = np.vstack((frame_input_x, frame_input_y))
                else:
                    # X and Y now correspond to rows and columns respectively
                    new_x = np.floor(frame_input_x / self.scale[1])
                    new_y = np.floor(frame_input_y / self.scale[0])
                    
                    new_ = np.vstack((new_x, new_y))
                    new_ = new_.astype(int)
                
                self.new_neg = np.take(new_, neg_track, axis=1)
                
                self.new_neg[0, :] *= -1
                
                pos_track = np.setdiff1d(np.arange(new_.shape[1]), neg_track)
                self.new_pos = np.take(new_, pos_track, axis=1)
                
                if self.version == 'naive':
                    self.naive()
                elif self.version == 'v1':
                    self.v1()
                elif self.version == 'v2':
                    self.v2()
                elif self.version == 'global':
                    self.spike_global()
                    
            if self.version == 'v1':
                sim_time, neuron_spike_times_pos, neuron_spike_times_neg = self.v1_finish()
                self.time = sim_time
                self.neuron_spike_times_pos = neuron_spike_times_pos
                self.neuron_spike_times_neg = neuron_spike_times_neg
                    
        return self.time, self.neuron_spike_times_pos, self.neuron_spike_times_neg

# new_resolution = np.array([20, 20])
# x = extractSourceSpikes('do-b_f_0.spikes', new_resolution, 'v1')
# time, neuron_spike_times_pos, neuron_spike_times_neg = x.read_input_spikes()