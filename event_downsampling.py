import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

# from tonic.slicers import slice_events_by_time

# events = np.load('dvs_struct_array.npy')

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

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]

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

def differentiator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, 
                              differentiator_time_bins: int = 2, noise_threshold: int = 0):
    """Downsample using an integrate-and-fire (I-F) neuron model with an additional differentiator 
    with a noise threshold similar to the membrane potential threshold in the I-F model. Multiply 
    x/y values by a spatial_factor obtained by dividing sensor size by the target size.
    
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        dt (float): step size for simulation, in ms.
        differentiator_time_bins (int): number of equally spaced time bins with respect to the dt 
                                        to be used for the differentiator.
        noise_threshold (int): number of events before a spike representing a new event is emitted.
        
    Returns:
        the down-sampled input events using the differentiator method.
    """
        
    assert "x" and "y" and "t" in events.dtype.names
    
    # Create time bins according to differentiator
    # events = time_bin_numpy(events, dt / differentiator_time_bins)
    
    # Call integrator method
    events_integrated = integrator_downsample(events=events, sensor_size=sensor_size, target_size=target_size, 
                                              dt=(dt / differentiator_time_bins), 
                                              noise_threshold=noise_threshold)
    
    # All event times
    events_adjusted = time_bin_numpy(events_integrated, dt)
    
    event_times = np.unique(events_adjusted["t"])
    
    # Separate by polarity
    events_positive = events_adjusted[events_adjusted["p"] == 1]
    events_negative = events_adjusted[events_adjusted["p"] == 0]
    
    frame_histogram = np.zeros((len(event_times), *np.flip(target_size), 2))
    
    for t, time in enumerate(event_times):
        xy_pos = events_positive[events_positive["t"] == time][["x", "y"]]
        xy_neg = events_negative[events_negative["t"] == time][["x", "y"]]
        
        frame_histogram[t,:,:,1] += np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0]
        frame_histogram[t,:,:,0] += np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0]
        
    # Differences between subsequent frames
    frame_differences = np.diff(frame_histogram, axis=0).clip(min=0)

    # Restructuring numpy array to structured array
    time_index, y_new, x_new, polarity_new = np.nonzero(frame_differences)
    
    events_new = np.column_stack((x_new, y_new, polarity_new, event_times[time_index]))
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)
    
def integrator_downsample(events: np.ndarray, sensor_size: tuple, target_size: tuple, dt: float, noise_threshold: int = 0):
    """Downsample using an integrate-and-fire (I-F) neuron model with a noise threshold similar to 
    the membrane potential threshold in the I-F model. Multiply x/y values by a spatial_factor 
    obtained by dividing sensor size by the target size.
    
    Parameters:
        events (ndarray): ndarray of shape [num_events, num_event_channels].
        sensor_size (tuple): a 3-tuple of x,y,p for sensor_size.
        target_size (tuple): a 2-tuple of x,y denoting new down-sampled size for events to be
                             re-scaled to (new_width, new_height).
        dt (float): temporal resolution of events in milliseconds.
        noise_threshold (int): number of events before a spike representing a new event is emitted.
        
    Returns:
        the down-sampled input events using the integrator method.
    """
        
    assert "x" and "y" and "t" in events.dtype.names
    
    events = events.copy()
    
    if np.issubdtype(np.int64, events["t"].dtype):
        dt *= 1000
    
    # Re-format event times to new temporal resolution
    events = time_bin_numpy(events, dt)
    # events_sliced = slice_events_by_time(events, time_window=0.5)
    
    # Downsample
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]
    
    # All event times
    event_times = np.unique(events["t"])
    
    # Separate by polarity
    events_positive = events[events["p"] == 1]
    events_negative = events[events["p"] == 0]
    
    # Running buffer of events in each pixel
    frame_spike = np.zeros(np.flip(target_size))
    
    events_new = []
    
    for time in event_times:
        xy_pos = events_positive[events_positive["t"] == time][["x", "y"]]
        xy_neg = events_negative[events_negative["t"] == time][["x", "y"]]
        
        # Sum in 2D space using histogram
        frame_histogram = np.subtract(np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0],
                                      np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0])
        
        frame_spike += frame_histogram
        
        coordinates_pos = np.stack(np.nonzero(np.maximum(frame_spike >= noise_threshold, 0))).T
        coordinates_neg = np.stack(np.nonzero(np.maximum(-frame_spike >= noise_threshold, 0))).T
        
        # Reset spiking coordinates to zero
        frame_spike[coordinates_pos] = 0
        frame_spike[coordinates_neg] = 0
        
        # Add to event buffer
        events_new.append(np.column_stack((coordinates_pos, np.ones((coordinates_pos.shape[0],1)), time*np.ones((coordinates_pos.shape[0],1)))))
        events_new.append(np.column_stack((coordinates_neg, np.zeros((coordinates_neg.shape[0],1)), time*np.ones((coordinates_neg.shape[0],1)))))
        
    events_new = np.concatenate(events_new.copy())
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)

# naive_downsample(events, sensor_size=(128,128), target_size=(20,20))