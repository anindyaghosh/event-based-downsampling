import argparse
import cv2
from dv import AedatFile
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--data_folder', type=str)
parser.add_argument('-i', '--input', type=str)

args = parser.parse_args()

data_folder = os.path.join(os.getcwd(), 'data', f'{args.data_folder}')

with AedatFile(f'{data_folder}/aedat/{args.input}.aedat4') as f:
    
    cams = ['Davis346', 'Dvxplorer']
    
    frames_array = []
    for e, event_input in enumerate(f.names):
        events = np.hstack([packet for packet in f[event_input].numpy()])
        
        # Access dimensions of the event stream
        height, width = f[event_input].size
    
        publisher_rate = 30 # fps
        
        # Correct for tsOffset
        events['timestamp'] -= events['timestamp'][0]
        
        # Save out events
        with open(f'{data_folder}/numpy/{args.input}_{cams[e]}.npy', 'wb') as event_file:
            np.save(event_file, events)
        
        # Temporally downsample to 1 ms resolution using publisher rate
        events['timestamp'] = (events['timestamp'] / (1/publisher_rate * 10**6)).astype(int)
        
        # Get total frames
        num_frames = events['timestamp'][-1] + 1
        
        frames = np.zeros((num_frames, height, width, 3))
        
        for t in range(num_frames):
            frame = np.zeros((height, width, 3))
            
            indices = np.where(events['timestamp'] == t)[0]
            sliced_events = np.take(events, indices)
            
            coordinates, polarity = np.vstack((sliced_events['x'], sliced_events['y'])), sliced_events['polarity']
            
            for p in range(2):
                polarity_indices = np.where(polarity == p)[0]
                x, y = np.take(coordinates, polarity_indices, axis=1)
                
                # Frames are in BGR format
                frame[...,p] = 255 * np.histogram2d(y, x, [np.arange(height+1), np.arange(width+1)])[0]
                
            frames[t] += frame
            
        frames_array.append(frames)
        
def visualise(cams):
    
    for f in frames_array:
        f[...,[1, 2]] = f[...,[2, 1]]
    
    for cam in cams:
        cv2.namedWindow(cam, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam, (640, 480))
    
    for n in range(num_frames):
        for c, cam in enumerate(cams):
            image = frames_array[c][n]
            resized = cv2.resize(image, (640, 480))
                
            cv2.putText(resized, cam, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.imshow(cam, resized.astype(np.uint8))
            
            cv2.waitKey(3)
    
    cv2.destroyAllWindows()
    
def videoOut(cams):
    output_dir = f'{data_folder}/video/{args.input}'
    os.makedirs(output_dir, exist_ok=True)
    
    for c, cam in enumerate(cams):
        out = cv2.VideoWriter(f'{output_dir}/{cam}.mp4', 
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              publisher_rate, 
                              (640, 480))
        for n in range(num_frames):
            image = frames_array[c][n]
            resized = cv2.resize(image, (640, 480))
                
            cv2.putText(resized, cam, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            out.write(resized.astype(np.uint8))
        out.release()
    
visualise(cams)
videoOut(cams)