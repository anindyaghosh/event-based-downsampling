import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured

from tonic.slicers import slice_events_by_time

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
    
    events = events.copy()
    
    # Call integrator method
    events_integrated = integrator_downsample(events, sensor_size=sensor_size, target_size=target_size, 
                                              dt=(dt / differentiator_time_bins), 
                                              noise_threshold=noise_threshold)
    
    if np.issubdtype(events["t"].dtype, np.integer):
        dt *= 1000
    
    # Reformat time to original dt
    events_adjusted = slice_events_by_time(events_integrated, time_window=dt)
    
    frame_histogram = np.zeros((len(events_adjusted), *np.flip(target_size), 2))
    
    for time, event in enumerate(events_adjusted):
        # To speed up algorithm
        if event.size >= 1:
            # Separate by polarity
            xy_pos = event[event["p"] == 1]
            xy_neg = event[event["p"] == 0]
            
            frame_histogram[time,:,:,1] += np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0]
            frame_histogram[time,:,:,0] += np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0]
            
    # Differences between subsequent frames
    frame_differences = np.diff(frame_histogram, axis=0).clip(min=0)

    # Restructuring numpy array to structured array
    time_index, y_new, x_new, polarity_new = np.nonzero(frame_differences)
    
    events_new = np.column_stack((x_new, y_new, polarity_new.astype(dtype=bool), time_index * dt))
    
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
    
    if np.issubdtype(events["t"].dtype, np.integer):
        dt *= 1000
    
    # Downsample
    spatial_factor = np.asarray(target_size) / sensor_size[:-1]

    events["x"] = events["x"] * spatial_factor[0]
    events["y"] = events["y"] * spatial_factor[1]
    
    # Re-format event times to new temporal resolution
    events_sliced = slice_events_by_time(events, time_window=dt)
    
    # Running buffer of events in each pixel
    frame_spike = np.zeros(np.flip(target_size))
    
    events_new = []
    
    for time, event in enumerate(events_sliced):
        # Separate by polarity
        xy_pos = event[event["p"] == 1]
        xy_neg = event[event["p"] == 0]
        
        # Sum in 2D space using histogram
        frame_histogram = np.subtract(np.histogram2d(xy_pos["y"], xy_pos["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0],
                                      np.histogram2d(xy_neg["y"], xy_neg["x"], [range(target_size[1] + 1), range(target_size[0] + 1)])[0])
        
        frame_spike += frame_histogram
            
        coordinates_pos = np.stack(np.nonzero(np.maximum(frame_spike >= noise_threshold, 0))).T
        coordinates_neg = np.stack(np.nonzero(np.maximum(-frame_spike >= noise_threshold, 0))).T
        
        # Reset spiking coordinates to zero
        frame_spike[coordinates_pos] = 0
        frame_spike[coordinates_neg] = 0
        
        # Restructure events
        events_new.append(np.column_stack((np.flip(coordinates_pos, axis=1), np.ones((coordinates_pos.shape[0],1)).astype(dtype=bool), 
                                           (time*dt)*np.ones((coordinates_pos.shape[0],1)))))
        
        events_new.append(np.column_stack((np.flip(coordinates_neg, axis=1), np.zeros((coordinates_neg.shape[0],1)).astype(dtype=bool), 
                                           (time*dt)*np.ones((coordinates_neg.shape[0],1)))))
        
    events_new = np.concatenate(events_new.copy())
    
    return unstructured_to_structured(events_new.copy(), dtype=events.dtype)