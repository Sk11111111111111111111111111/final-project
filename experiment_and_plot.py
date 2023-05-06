import os
import random as rn
import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend
KU = keras.utils
from config import Config
c = Config()
from model_keras import Model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from pathlib import Path
import pandas as pd
import numpy as np
import utils

# Set seeds
utils.set_global_seed(c.SEED, use_parallelism=c.USE_PARALLELISM)

# Download the MNIST dataset
# X_train.shape = (60000, 28, 28)
# y_train.shape = (60000,) (the elements are the actual labels)
# X_test.shape = (10000, 28, 28)
# y_test.shape = (10000,)
MNIST = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = MNIST.load_data()

# Preprocess the data (reshape, rescale, etc.)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Preprocess class labels
y_train = KU.to_categorical(y_train, num_classes=10)
y_test = KU.to_categorical(y_test, num_classes=10)

# Split into train/val sets
split_idx = int(c.VAL_SPLIT * X_train.shape[0])
X_val = X_train[:split_idx]
y_val = y_train[:split_idx]
X_train = X_train[split_idx:]
y_train = y_train[split_idx:]

# Create model from config
M = Model(c)
M.load()

# Run prediction method
#M.predict(X_val)
for name, metric in zip(M.model.metrics_names, M.model.evaluate(x=X_test, y=y_test, batch_size=128)):
    print("{} = {}".format(name, metric))

class Experiment(object):
    """ Convenience class for containing all the information about an experiment. """
    def __init__(self, config, logs):
        self.config = config    # Config class associated with the experiment
        self.logs = logs        # File path to log.csv (this is a Path class)
        self.x = None           # Values of quantities to plot in Youden plot
        self.y = None

    def get_config(self):
        return self.config

    def get_logs(self):
        return pd.read_csv(self.logs)

def gather_experiments(path="./Experiments"):

    experiments = []

    # Get all the experiment directories
    dirs = Path().glob(path+"/*")
    # Iterate over each experiment directory
    for exp_dir in dirs:
        # Extract config file and training logs
        config = list(exp_dir.glob("config*.py"))
        logs = list(exp_dir.glob("logs/log.csv"))
        # Create Experiment object from config and logs
        assert (len(config) == 1), "Unexpected number of config files"
        assert (len(logs) == 1), "Unexpected number of log files"
        config = config[0]; logs = logs[0]
        # Import the Config class from the config file
        config_module = __import__(".".join(config.parts)[:-3], globals(), locals(), -1)
        config = config_module.Config()
        exp = Experiment(config, logs)
        experiments.append(exp)
    return experiments

def youden_plot(q1, q2, plot_scale="linear"):
    """
    Generates a Youden plot of all data. Separates experiments by theta schedule and learning rate schedule.
    :param q1: string; name of quantity to plot on x axis
    :param q2: string; name of quantity to plot on y axis
    :param base_or_exp: string; whether to analyze baseline or experimental data
    :return:
    """
    # Gather all experiments
    experiments = []
    #experiments.extend(gather_experiments(path="./Experiments/Baselines"))
    #experiments.extend(gather_experiments(path="./Experiments/Experiments/ManualTuning"))
    experiments.extend(gather_experiments(path="./Experiments/Experiments/GridSearch"))
    experiments.sort(key=lambda e: e.config.NAME)

    # Extract relevant data from logs
    # Iterate through each experiment
    for exp in experiments:
        logs = exp.get_logs()
        # For each experiment, get the values corresponding to q1 and q2
        for q, attr in zip([q1, q2], ["x", "y"]):
            invert_flag = False
            # Split the argument into pieces: mod(ifier), metric, and train/val
            tags = q.lower().split(" ")
            if len(tags) == 2: mod, metric = tags; train_val = "train"
            elif len(tags) == 3: mod, train_val, metric = tags
            else: raise ValueError("Unsupported quantity requested for Youden plot")

            # Determine whether to invert the metric (like you measured accuracy but want error)
            if metric in ["err", "error"]:
                invert_flag = True

            # Determine metric
            metric = metric.replace("_", "")
            if metric in ["acc", "accuracy", "err", "error"]: metric = "categorical_accuracy"
            elif metric in ["f1", "fbeta", "fbetamacro"]: metric = "fbeta_macro"
            elif metric in ["lr", "learningrate"]: metric = "learning_rate"
            elif metric in ["loss"]: metric = "loss"
            elif metric in ["prec", "precision", "precision_macro"]: metric = "precision_macro"
            elif metric in ["recall", "recallmacro"]: metric = "recall_macro"
            elif metric in ["time", "traintime"]: metric = "train_time"
            else: raise ValueError("Unsupported quantity requested for Youden plot")

            # Determine whether train or validation
            if train_val == "train": pass
            elif train_val in ["val", "validation"]: metric = "val_" + metric
            else: raise ValueError("Must specify train or validation quantity")

            # Compute value depending on modifier and inverse_flag
            if mod in ["fin", "final"]: val = logs[metric].iloc[-1]
            elif mod in ["max", "maximum"]:
                if invert_flag: val = logs[metric].min()
                else: val = logs[metric].max()
            elif mod in ["min", "minimum"]:
                if invert_flag: val = logs[metric].max()
                else: val = logs[metric].min()
            elif mod in ["avg", "average", "ave", "mean"]: val = logs[metric].mean()
            else: raise ValueError("Must specify metric modifier in Youden plot")

            if invert_flag: setattr(exp, attr, 1-val)
            else: setattr(exp, attr, val)

            if val <= 0: print("less than zero error for {}".format(exp.config.NAME))

    # Plot data
    theta_schedule_color_map = {"constant": "red", "linear": "green", "sqrt": "orange", "pow": "blue"}
    plt.figure("youden")
    plt.xscale(plot_scale)
    plt.yscale(plot_scale)
    minx, miny = np.inf, np.inf
    maxx, maxy = 0, 0
    for i, exp in enumerate(experiments):
        if exp.x < minx: minx = exp.x
        if exp.x > maxx: maxx = exp.x
        if exp.y < miny: miny = exp.y
        if exp.y > maxy: maxy = exp.y
        plt.scatter(exp.x, exp.y,
                    c=theta_schedule_color_map[exp.config.THETA_DECAY],
                    marker="${}$".format(i),
                    s=40*len(str(i)))
        print("{}:\t{}".format(i, exp.config.NAME))
        print("\t\tInitial LR: {:.1e}, regularization: {} - {:.1e}".format(exp.config.INITIAL_LR,
                                                               exp.config.REGULARIZER,
                                                               exp.config.REGULARIZATION_COEFFICIENT))
    plt.plot([max(minx,miny), min(maxx,maxy)], [max(minx,miny), min(maxx,maxy)], "k--")
    plt.xlabel(q1)
    plt.ylabel(q2)
    plt.xlim(xmin=minx*(1/1.2), xmax=maxx*(1.2))
    plt.ylim(ymin=miny*(1/1.2), ymax=maxy*(1.2))
    handles = [mpatch.Patch(color=val, label=key) for key, val in theta_schedule_color_map.items()]
    plt.legend(handles=handles, loc="best", title="Theta schedule")
    plt.tight_layout()

    # Draw/show plot
    #plt.show()
    plt.savefig("youden.png")