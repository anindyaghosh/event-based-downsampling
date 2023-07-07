import time
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

events = np.load('dvs_struct_array.npy')

def naive_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple):
    """Downsample the classic "naive" Tonic way. Multiply x/y values by a spatial_factor 
    obtained by dividing sensor size by the target size.
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
    Returns:
        the downsampled input events.
    """
    
    assert "x" and "y" in events.dtype.names
    
    events = events.copy()
    
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[1]
    events["y"] = events["y"] * spatial_factor[0]

    return events

def time_bin_numpy(events: np.ndarray, time_bin_interval: float):
    """Temporally downsample the events into discrete time bins as stipulated by time_bin_intervals.
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        time_bin_interval (float): time bin size for events e.g. every 0.5 ms: time_bin_interval = 0.5.
    Returns:
        the input events with rewritten timestamps.
    """
    
    events = events.copy()
    reciprocal_interval = 1 / time_bin_interval
    events["t"] = np.round(events["t"] * reciprocal_interval, 0) / reciprocal_interval
    
    return events

def diff_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, noise_threshold: int = 0):
    """Downsample using an integrate-and-fire (I-F) neuron model with a noise threshold similar to 
    the membrane potential threshold in the I-F model. Multiply x/y values by a spatial_factor 
    obtained by dividing sensor size by the target size.
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
    Returns:
        the downsampled input events using the differentiator method.
    """
        
    assert "x" and "y" and "t" in events.dtype.names
    
    events = events.copy()
    
    # Downsample
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[1]
    events["y"] = events["y"] * spatial_factor[0]
    
    # All event times
    event_times = np.unique(events["t"])
    
    # Separate by polarity
    events_positive = events[events["p"] == 1]
    events_negative = events[events["p"] == 0]
    
    # Running buffer of events in each pixel
    frame_spike = np.zeros(target_size)
    
    events_new = []
    
    for time in event_times:
        xy_pos = events_positive[events_positive["t"] == time][["x", "y"]]
        xy_neg = events_negative[events_negative["t"] == time][["x", "y"]]
        
        frame_histogram = np.subtract(np.histogram2d(xy_pos["x"], xy_pos["y"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0],
                                      np.histogram2d(xy_neg["x"], xy_neg["y"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0])
        
        frame_spike += frame_histogram
        
        coordinates_pos = np.stack(np.nonzero(np.maximum(frame_spike > noise_threshold, 0))).T
        coordinates_neg = np.stack(np.nonzero(np.maximum(-frame_spike > noise_threshold, 0))).T
        
        # Reset spiking coordinates to zero
        frame_spike[coordinates_pos] = 0
        frame_spike[coordinates_neg] = 0
        
        # Add to event buffer
        events_new.append(np.column_stack((coordinates_pos, np.ones((coordinates_pos.shape[0],1)), time*np.ones((coordinates_pos.shape[0],1)))))
        events_new.append(np.column_stack((coordinates_neg, np.zeros((coordinates_neg.shape[0],1)), time*np.ones((coordinates_neg.shape[0],1)))))
        
    events_new = np.concatenate(events_new.copy())
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)

# Time bin of 1 ms
events = time_bin_numpy(events, 1.0)
t1 = time.perf_counter()
events_diff = diff_downsample(events, (128, 128, 2), (16, 16), noise_threshold=20)
t2 = time.perf_counter()
print(f'{t2-t1:.2f}')