import numpy as np
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))

class txt2spike:
    def __init__(self, filename, version, polarity):
        self.filename = filename
        self.version = version
        self.polarity = polarity
        
    def resolution_check(self):
        if 'downsampled' in self.filename:
            self.resolution = np.array([64, 36])
        else:
            self.resolution = np.array([640, 360])
            
    def polarity_check(self):
        if self.version == 'naive':
            if self.polarity not in ['+', '-']:
                sys.exit('Invalid polarity')
        else:
            self.polarity = ''

    def txt_convert(self):
        self.resolution_check()
        self.polarity_check()
        
        timestamps = []
        events_x = []
        events_y = []
        polarities = []
        
        if self.version == 'v1':
            self.timesteps_per_frame = 33
        else:
            self.timesteps_per_frame = 1

        with open (self.filename, 'r') as file:
            for line in file:
                time, y, x, polarity = line.split(' ')
                
                # Timestamp in ms from rostime ns
                if self.version == 'v1':
                    timestamps.append(int((int(time) / 10**3) / (1000 / self.timesteps_per_frame)))
                else:
                    timestamps.append(int(int(time) / 10**6))
                    
                events_x.append(x)
                events_y.append(y)
                polarities.append(polarity)
                
            timestamps = np.asarray(timestamps, dtype=int)
            events_x = np.asarray(events_x, dtype=int)
            events_y = np.asarray(events_y, dtype=int)
            polarities = np.asarray(polarities, dtype=int)

        pix = np.ravel_multi_index([events_x, events_y], (self.resolution[1], self.resolution[0]))
        
        if self.version == 'v1':
            spike_list = [[] for i in np.arange(max(timestamps) + 1) / ((1000 / self.timesteps_per_frame))]
        else:
            spike_list = [[] for i in np.arange(max(timestamps) + 1)]
        
        if self.version == 'naive':
        # **YUCK!**
        # Needed for old format of writing out spike files
            for i, t in enumerate(timestamps):
                if self.polarity == '+':
                    
                    if polarities[i].any():
                        spike_list[t].append(pix[i])
                    else:
                        pass
                elif self.polarity == '-':
                    if polarities[i].any():
                        pass
                    else:
                        spike_list[t].append(pix[i])
        else:
            for i, t in enumerate(timestamps):
                if polarities[i].any():
                    pass
                else:
                    pix[i] *= -1
                spike_list[t].append(pix[i])
                
        spike_list = np.asarray(spike_list, dtype=object)
        
        return spike_list
    
    def writeSpikeFile(self, spike_list):
        
        polarity_addendum = ''
        
        if self.version == 'naive':
            polarity_folder = '../do_dataset/polarity'
            polarity_addendum = '_' + self.polarity
        elif self.version == 'v1':
            polarity_folder = '../do_dataset/polarity_add33'
        elif self.version in ['v2', 'global']:
            polarity_folder = '../do_dataset/polarity_add'
            
        filehead, _ = os.path.basename(self.filename).split('.')
        path_to_folder = os.path.join(root, polarity_folder, filehead + polarity_addendum + '.spikes')
        
        with open(path_to_folder, 'w') as file:
            for i, lst in enumerate(spike_list):
                if spike_list[i]:
                    string_ints = [str(int) for int in spike_list[i]]
                    str_of_ints = ",".join(string_ints)
                    lines = ''.join([str(i) + ';' + str_of_ints])
                    file.write(''.join([lines + '\n']))
    
def getAllTxtFiles():
    
    esim_files = []
    
    subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
    for folder in subfolders:
        files = os.listdir(folder)
        for file in files:
            if file.endswith('.txt'):
                filePath = os.path.join(root, folder, file)
                esim_files.append(filePath)
            else:
                pass
            
    return esim_files

def action(file, version, polarity):
    txtfile = txt2spike(file, version, polarity)
    spike_list = txtfile.txt_convert()
    txtfile.writeSpikeFile(spike_list)

version = 'v1'

esim_files = getAllTxtFiles()
for file in esim_files:
    if version in ['global', 'v2']:
        action(file, version, '+')
    else:
        if 'downsampled' not in file:
            action(file, version, '+')