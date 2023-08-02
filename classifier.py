import csv
import os
import numpy as np

from argparse import ArgumentParser
from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL
from ml_genn import Connection, Population, Network
from ml_genn.callbacks import Callback, Checkpoint
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import (AdaptiveLeakyIntegrateFire, LeakyIntegrate,
                             LeakyIntegrateFire, SpikeInput)
from ml_genn.serialisers import Numpy

from glob import glob
from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data, preprocess_tonic_spikes)

import event_downsampling as event_ds


class CSVTrainLog(Callback):
    def __init__(self, filename, output_pop, resume):
        # Create CSV writer
        self.file = open(filename, "a" if resume else "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")

        # Write header row if we're not resuming from an existing training run
        if not resume:
            self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])

        self.output_pop = output_pop

    def on_epoch_begin(self, epoch):
        self.start_time = perf_counter()

    def on_epoch_end(self, epoch, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class CSVTestLog(Callback):
    def __init__(self, filename, epoch, resume, output_pop):
        # Create CSV writer
        self.file = open(filename, "a" if resume else "w")
        self.csv_writer = csv.writer(self.file, delimiter=",")
        if not resume:
            self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "Time"])
        self.epoch = epoch
        self.output_pop = output_pop

    def on_test_begin(self):
        self.start_time = perf_counter()

    def on_test_end(self, metrics):
        m = metrics[self.output_pop]
        self.csv_writer.writerow([self.epoch, m.total, m.correct, 
                                  perf_counter() - self.start_time])
        self.file.flush()

class ConnectivityCheckpoint(Callback):
    def __init__(self, serialiser="numpy"):
        self.serialiser = serialiser

    def set_params(self, compiled_network, **kwargs):
        # Extract compiled network
        self._compiled_network = compiled_network

    def on_epoch_end(self, epoch, metrics):
        # If we should checkpoint this epoch
        self._compiled_network.save_connectivity((epoch,), self.serialiser)



def pad_hidden_layer_argument(arg, num_hidden_layers, context, default=None):
    # If argument wasn't specified but there is a default, repeat default for each hidden layer
    if arg is None and default is not None:
        return [default] * num_hidden_layers
    elif len(arg) == 1:
        return arg * num_hidden_layers
    elif len(arg) != num_hidden_layers:
        raise RuntimeError(f"{context} either needs to be specified as a single "
                           f" value or for each {num_hidden_layers} layers")
    else:
        return arg

def inference(genn_kwargs, args, network, serialiser, latest_spike_time, epoch, resume):
    # Load network state from final checkpoint
    network.load((epoch,), serialiser)

    compiler = InferenceCompiler(evaluate_timesteps=int(np.ceil(latest_spike_time)),
                                 batch_size=1 if args.cpu else args.batch_size, rng_seed=args.seed,
                                 reset_vars_between_batches=False, 
                                 kernel_profiling=args.kernel_profiling, **genn_kwargs)
    compiled_net = compiler.compile(network, name=f"classifier_test_{unique_suffix}")

    with compiled_net:
        # Perform warmup evaluation
        # **TODO** subset of data
        compiled_net.evaluate({input: spikes},
                              {output: labels})

        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar",
                     CSVTestLog(f"test_output_{unique_suffix}.csv", epoch, resume, output)]
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels},
                                            callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
        
        if args.kernel_profiling:
            print(f"Neuron update time = {compiled_net.genn_model.neuron_update_time}")
            print(f"Presynaptic update time = {compiled_net.genn_model.presynaptic_update_time}")
            print(f"Reset time = {compiled_net.genn_model.get_custom_update_time('Reset')}")

parser = ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
parser.add_argument("--train", action="store_true", help="Train model")
parser.add_argument("--cpu", action="store_true", help="Use CPU for inference")
parser.add_argument("--kernel-profiling", action="store_true", help="Output kernel profiling data")
parser.add_argument("--test-all", action="store_true", help="Test all checkpoints up to num epochs")
parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--dataset", choices=["smnist", "shd", "dvs_gesture", "mnist"], required=True)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--resume-epoch", type=int, default=None)

parser.add_argument("--hidden-size", type=int, nargs="*")
parser.add_argument("--hidden-recurrent", choices=["True", "False"], nargs="*")
parser.add_argument("--hidden-model", choices=["lif", "alif"], nargs="*")
parser.add_argument("--hidden-input-sparsity", type=float, nargs="*")
parser.add_argument("--hidden-recurrent-sparsity", type=float, nargs="*")

parser.add_argument("--downsampling-method", type=str, required=True)
parser.add_argument("--target-resolution", type=int, required=True)

args = parser.parse_args(["--train", "--seed", "2345", "--dataset", "dvs_gesture",
                          "--num-epochs", "100", "--hidden-size", "256", "256",
                          "--hidden-recurrent", "False", "True", "--hidden-model",
                          "alif", "alif", "--hidden-input-sparsity", "0.1", "0.1",
                          "--hidden-recurrent-sparsity", "0.01", "--downsampling-method",
                          "differentiator", "--target-resolution", "8"])

# args = parser.parse_args()

num_hidden_layers = max(len(args.hidden_size), 
                        len(args.hidden_recurrent),
                        len(args.hidden_model))
print(f"{num_hidden_layers} hidden layers")

# Pad hidden layer arguments
args.hidden_size = pad_hidden_layer_argument(args.hidden_size, 
                                             num_hidden_layers,
                                             "Hidden layer size")
args.hidden_recurrent = pad_hidden_layer_argument(args.hidden_recurrent, 
                                                  num_hidden_layers,
                                                  "Hidden layer recurrentness")
args.hidden_model = pad_hidden_layer_argument(args.hidden_model, 
                                              num_hidden_layers,
                                              "Hidden layer neuron model")
args.hidden_input_sparsity = pad_hidden_layer_argument(args.hidden_input_sparsity, 
                                                       num_hidden_layers,
                                                       "Hidden layer input sparsity",
                                                       1.0)
args.hidden_recurrent_sparsity = pad_hidden_layer_argument(args.hidden_recurrent_sparsity, 
                                                          num_hidden_layers,
                                                          "Hidden layer recurrent sparsity",
                                                          1.0)
# Figure out unique suffix for model data
# unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         # else str(val))
                         # for arg, val in vars(args).items()
                         # if arg not in ["train", "cpu", "resume_epoch",
                                        # "test_all", "kernel_profiling"])

# If dataset is MNIST
spikes = []
labels = []
num_input = None
num_output = None
if args.dataset == "mnist":
    import mnist

    # Latency encode MNIST digits
    num_input = 28 * 28
    num_output = 10
    labels = mnist.train_labels() if args.train else mnist.test_labels()
    spikes = log_latency_encode_data(
        mnist.train_images() if args.train else mnist.test_images(),
        20.0, 51)
# Otherwise
else:
    from tonic.datasets import DVSGesture, SHD, SMNIST
    from tonic.transforms import Compose, Downsample

    # Load Tonic datasets
    if args.dataset == "shd":
        dataset = SHD(save_to='./data', train=args.train)
        sensor_size = dataset.sensor_size
    elif args.dataset == "smnist":
        dataset = SMNIST(save_to='./data', train=args.train, 
                                        duplicate=False, num_neurons=79)
        sensor_size = dataset.sensor_size
    elif args.dataset == "dvs_gesture":
        
        # transform = Compose([Downsample(spatial_factor=0.25)])
        # dataset = DVSGesture(save_to='./data', train=args.train, transform=transform)
        
        dataset = DVSGesture(save_to='./data', train=args.train)
        sensor_size = (args.target_resolution, args.target_resolution, 2)

    # Get number of input and output neurons from dataset 
    # and round up outputs to power-of-two
    num_input = int(np.prod(sensor_size))
    num_output = len(dataset.classes)

    # Preprocess spike
    num_events = 0
    for events, label in dataset:
        if args.downsampling_method == 'naive':
            events = event_ds.naive_downsample(events, sensor_size=(128, 128, 2), 
                                               target_size=sensor_size[:-1])
        elif args.downsampling_method == 'integrator':
            events = event_ds.integrator_downsample(events, sensor_size=(128, 128, 2), 
                                                    target_size=sensor_size[:-1],
                                                    dt=0.05, noise_threshold=2)
        elif args.downsampling_method == 'differentiator':
            # t1 = perf_counter()
            events = event_ds.differentiator_downsample(events, sensor_size=(128, 128, 2), 
                                                        target_size=sensor_size[:-1],
                                                        dt=0.05, differentiator_time_bins=3, 
                                                        noise_threshold=2)
            # t2 = perf_counter()
            # print(f'{t2-t1:.2f}')
            # import pdb
            # pdb.set_trace()
        
        num_events += len(events)
        
        spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                              sensor_size))
        labels.append(label)
    print(f"Total spikes: {num_events}")
    unique_suffix = f"{args.downsampling_method}_{str(args.target_resolution)}"

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

genn_kwargs = {"selectGPUByDeviceID": True,
               "deviceSelectMethod": DeviceSelect_MANUAL,
               "manualDeviceID": args.device_id}

serialiser = Numpy("checkpoints_" + unique_suffix)
network = Network()
with network:
    # Add spike input population
    input = Population(SpikeInput(max_spikes=args.batch_size * max_spikes),
                       num_input)
    
    # Add output population
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var", softmax=args.train),
                        num_output)

    # Loop through hidden layers
    hidden = []
    for i, (s, r, m, in_sp, rec_sp) in enumerate(zip(args.hidden_size, 
                                                 args.hidden_recurrent,
                                                 args.hidden_model,
                                                 args.hidden_input_sparsity,
                                                 args.hidden_recurrent_sparsity)):
        # Add population
        if m == "alif":
            hidden.append(Population(AdaptiveLeakyIntegrateFire(v_thresh=0.6,
                                                                tau_refrac=5.0,
                                                                relative_reset=True,
                                                                integrate_during_refrac=True),
                                     s))
        else:
            hidden.append(Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                                        tau_refrac=5.0,
                                                        relative_reset=True,
                                                        integrate_during_refrac=True),
                                     s))

        # If recurrent, add recurrent connections
        if r == "True":
            rec_weight = Normal(sd=1.0 / np.sqrt(s)) 
            rec_connectivity = (Dense(rec_weight) if rec_sp == 1.0 
                                else FixedProbability(rec_sp, rec_weight))
            Connection(hidden[-1], hidden[-1], rec_connectivity)

        # Add connection to output layer
        Connection(hidden[-1], output, Dense(Normal(sd=1.0 / np.sqrt(hidden[-1].shape[0]))))

        # If this is first hidden layer, add input connections
        if i == 0:
            in_weight = Normal(sd=1.0 / np.sqrt(num_input))
            in_connectivity = (Dense(in_weight) if in_sp == 1.0 
                               else FixedProbability(in_sp, in_weight, True))
            Connection(input, hidden[-1], in_connectivity)
        # Otherwise, add connection to previous hidden layer
        else:
            in_weight = Normal(sd=1.0 / np.sqrt(hidden[-2].shape[0]))
            in_connectivity = (Dense(in_weight) if in_sp == 1.0 
                               else FixedProbability(in_sp, in_weight))
            Connection(hidden[-2], hidden[-1], in_connectivity)

# If we're training model
if args.train:
    assert not args.cpu

    # If we should resume traing from a checkpoint, load checkpoint
    if args.resume_epoch is not None:
        network.load((args.resume_epoch,), serialiser)

    # Create EProp compiler and compile
    compiler = EPropCompiler(example_timesteps=int(np.ceil(latest_spike_time)),
                             losses="sparse_categorical_crossentropy", rng_seed=args.seed,
                             optimiser="adam", batch_size=args.batch_size, 
                             kernel_profiling=args.kernel_profiling, **genn_kwargs)
    compiled_net = compiler.compile(network, name=f"classifier_train_{unique_suffix}")

    with compiled_net:
        # Evaluate model on SHD
        start_time = perf_counter()
        start_epoch = 0 if args.resume_epoch is None else (args.resume_epoch + 1)
        callbacks = ["batch_progress_bar", Checkpoint(serialiser),
                     CSVTrainLog(f"train_output_{unique_suffix}.csv", output,
                                 args.resume_epoch is not None),
                     ConnectivityCheckpoint(serialiser)]
        metrics, _  = compiled_net.train({input: spikes},
                                         {output: labels},
                                         num_epochs=args.num_epochs,
                                         callbacks=callbacks, shuffle=True,
                                         start_epoch=start_epoch)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
else:
    print(f"Loading inference model from checkpoint {args.num_epochs - 1}")

    # Use CPU backend if desired
    if args.cpu:
        genn_kwargs["backend"]="SingleThreadedCPU"

    # Loop through trained epochs
    if args.test_all:
        for e in range(args.num_epochs):
             inference(genn_kwargs, args, network, serialiser, latest_spike_time, e, e > 0)
    else:
        inference(genn_kwargs, args, network, serialiser, latest_spike_time, args.num_epochs - 1, False)
