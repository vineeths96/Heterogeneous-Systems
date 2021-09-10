import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    TransformedBbox,
    BboxPatch,
    BboxConnector,
)


NUM_REPEATS = 5


def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


def plot_loss_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            dynamic_partition = None
            enhance = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("dynamic_partition"):
                        dynamic_partition = line.split(": ")[-1] == "True"

                    if line.startswith("enhance"):
                        enhance = line.split(": ")[-1] == "True"

            if dynamic_partition and enhance:
                label = "EDP-SGD"
            elif dynamic_partition:
                label = "DP-SGD"
            else:
                label = "Sync-SGD"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("train_loss")
            num_epochs = loss.shape[1]

            mean_loss = np.mean(loss, axis=0)
            std_dev_loss = np.std(loss, axis=0)

            axes_main.plot(np.arange(num_epochs), mean_loss, label=label)
            axes_main.fill_between(
                np.arange(num_epochs),
                mean_loss - std_dev_loss,
                mean_loss + std_dev_loss,
                alpha=0.25,
            )
            axes_main.set_ylim(0, 2.5)

            # axes_inner.plot(axes_inner_range, mean_loss[axes_inner_range])
            # axes_inner.fill_between(
            #     axes_inner_range,
            #     mean_loss[axes_inner_range] - std_dev_loss[axes_inner_range],
            #     mean_loss[axes_inner_range] + std_dev_loss[axes_inner_range],
            #     alpha=0,
            # )
            # axes_inner.set_ylim(0, 2.5)

            # axes_main.plot(loss, label=label)
            # axes_inner.plot(axes_inner_range, mean_loss[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Loss")
        # axes_main.set_title(f"Loss curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_{models[group_ind]}.svg")
        plt.show()


def plot_loss_time_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            dynamic_partition = None
            enhance = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("dynamic_partition"):
                        dynamic_partition = line.split(": ")[-1] == "True"

                    if line.startswith("enhance"):
                        enhance = line.split(": ")[-1] == "True"

            if dynamic_partition and enhance:
                label = "EDP-SGD"
            elif dynamic_partition:
                label = "DP-SGD"
            else:
                label = "Sync-SGD"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            loss = log_dict[()].get("train_loss")
            time = log_dict[()].get("time")

            num_epochs = loss.shape[1]

            for i in reversed(range(1, NUM_REPEATS)):
                time[i] -= time[i - 1][-1]

            mean_loss = np.mean(loss, axis=0)
            std_dev_loss = np.std(loss, axis=0)
            time = np.mean(time, axis=0)

            axes_main.plot(time, mean_loss, label=label)
            axes_main.fill_between(
                time,
                mean_loss - std_dev_loss,
                mean_loss + std_dev_loss,
                alpha=0.25,
            )
            axes_main.set_ylim(0, 2.5)

            # axes_inner.plot(time[axes_inner_range], mean_loss[axes_inner_range])
            # axes_inner.fill_between(
            #     time[axes_inner_range],
            #     mean_loss[axes_inner_range] - std_dev_loss[axes_inner_range],
            #     mean_loss[axes_inner_range] + std_dev_loss[axes_inner_range],
            #     alpha=0,
            # )
            # axes_inner.set_ylim(0, 2.5)

            # axes_main.plot(time, loss, label=label)
            # axes_inner.plot(time[axes_inner_range], loss[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Time (sec)")
        axes_main.set_ylabel("Loss")
        # axes_main.set_title(f"Loss Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/loss_time_{models[group_ind]}.svg")
        plt.show()


def plot_top1_accuracy_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            dynamic_partition = None
            enhance = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("dynamic_partition"):
                        dynamic_partition = line.split(": ")[-1] == "True"

                    if line.startswith("enhance"):
                        enhance = line.split(": ")[-1] == "True"

            if dynamic_partition and enhance:
                label = "EDP-SGD"
            elif dynamic_partition:
                label = "DP-SGD"
            else:
                label = "Sync-SGD"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy") * 100
            num_epochs = top1_accuracy.shape[1]

            mean_top1_accuracy = np.mean(top1_accuracy, axis=0)
            std_dev_top1_accuracy = np.std(top1_accuracy, axis=0)

            axes_main.plot(np.arange(num_epochs), mean_top1_accuracy, label=label)
            axes_main.fill_between(
                np.arange(num_epochs),
                mean_top1_accuracy - std_dev_top1_accuracy,
                mean_top1_accuracy + std_dev_top1_accuracy,
                alpha=0.25,
            )

            # axes_inner.plot(axes_inner_range, mean_top1_accuracy[axes_inner_range])
            # axes_inner.fill_between(
            #     axes_inner_range,
            #     mean_top1_accuracy[axes_inner_range] - std_dev_top1_accuracy[axes_inner_range],
            #     mean_top1_accuracy[axes_inner_range] + std_dev_top1_accuracy[axes_inner_range],
            #     alpha=0,
            # )

            # axes_main.plot(top1_accuracy, label=label)
            # axes_inner.plot(axes_inner_range, top1_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_{models[group_ind]}.svg")
        plt.show()


def plot_top5_accuracy_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            dynamic_partition = None
            enhance = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("dynamic_partition"):
                        dynamic_partition = line.split(": ")[-1] == "True"

                    if line.startswith("enhance"):
                        enhance = line.split(": ")[-1] == "True"

            if dynamic_partition and enhance:
                label = "EDP-SGD"
            elif dynamic_partition:
                label = "DP-SGD"
            else:
                label = "Sync-SGD"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top5_accuracy = log_dict[()].get("test_top5_accuracy") * 100
            num_epochs = top5_accuracy.shape[1]

            mean_top5_accuracy = np.mean(top5_accuracy, axis=0)
            std_dev_top5_accuracy = np.std(top5_accuracy, axis=0)

            axes_main.plot(np.arange(num_epochs), mean_top5_accuracy, label=label)
            axes_main.fill_between(
                np.arange(num_epochs),
                mean_top5_accuracy - std_dev_top5_accuracy,
                mean_top5_accuracy + std_dev_top5_accuracy,
                alpha=0.25,
            )

            # axes_inner.plot(axes_inner_range, mean_top5_accuracy[axes_inner_range])
            # axes_inner.fill_between(
            #     axes_inner_range,
            #     mean_top5_accuracy[axes_inner_range] - std_dev_top5_accuracy[axes_inner_range],
            #     mean_top5_accuracy[axes_inner_range] + std_dev_top5_accuracy[axes_inner_range],
            #     alpha=0,
            # )

            # axes_main.plot(top5_accuracy, label=label)
            # axes_inner.plot(axes_inner_range, top5_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Epochs")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top5_{models[group_ind]}.svg")
        plt.show()


def plot_top1_accuracy_time_curves(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            dynamic_partition = None
            enhance = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("dynamic_partition"):
                        dynamic_partition = line.split(": ")[-1] == "True"

                    if line.startswith("enhance"):
                        enhance = line.split(": ")[-1] == "True"

            if dynamic_partition and enhance:
                label = "EDP-SGD"
            elif dynamic_partition:
                label = "DP-SGD"
            else:
                label = "Sync-SGD"

            log_dict = np.load(os.path.join(experiment, "log_dict.npy"), allow_pickle=True)
            top1_accuracy = log_dict[()].get("test_top1_accuracy") * 100
            time = log_dict[()].get("time")

            num_epochs = top1_accuracy.shape[1]

            for i in reversed(range(1, NUM_REPEATS)):
                time[i] -= time[i - 1][-1]

            mean_top1_accuracy = np.mean(top1_accuracy, axis=0)
            std_dev_top1_accuracy = np.std(top1_accuracy, axis=0)
            time = np.mean(time, axis=0)

            axes_main.plot(time, mean_top1_accuracy, label=label)
            axes_main.fill_between(
                time,
                mean_top1_accuracy - std_dev_top1_accuracy,
                mean_top1_accuracy + std_dev_top1_accuracy,
                alpha=0.25,
            )

            # axes_inner.plot(time[axes_inner_range], mean_top1_accuracy[axes_inner_range])
            # axes_inner.fill_between(
            #     time[axes_inner_range],
            #     mean_top1_accuracy[axes_inner_range] - std_dev_top1_accuracy[axes_inner_range],
            #     mean_top1_accuracy[axes_inner_range] + std_dev_top1_accuracy[axes_inner_range],
            #     alpha=0,
            # )

            # axes_main.plot(time, top1_accuracy, label=label)
            # axes_inner.plot(time[axes_inner_range], top1_accuracy[axes_inner_range])

        # axes_inner.grid()
        # mark_inset(
        #     axes_main,
        #     axes_inner,
        #     loc1a=1,
        #     loc1b=1,
        #     loc2a=2,
        #     loc2b=2,
        #     fc="none",
        #     ec="0.5",
        # )

        # axes_main.grid()
        axes_main.set_xlabel("Time (sec)")
        axes_main.set_ylabel("Test Accuracy (%)")
        # axes_main.set_title(f"Accuracy Time curve {models[group_ind]}")
        axes_main.legend()

        plt.tight_layout()
        plt.savefig(f"./plots/top1_time_{models[group_ind]}.svg")
        plt.show()


def plot_process_times_histogram(log_path):
    models = ["ResNet50", "VGG16"]
    experiment_groups = [glob.glob(f"{log_path}/*{model}") for model in models]

    for group_ind, experiment_group in enumerate(experiment_groups):
        fig, axes_main = plt.subplots()
        # axes_inner = plt.axes([0.65, 0.4, 0.3, 0.3])
        # axes_inner_range = list(range(0, 50))

        experiment_group.sort()

        for ind, experiment in enumerate(experiment_group):
            dynamic_partition = None
            enhance = None

            with open(os.path.join(experiment, "success.txt")) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("dynamic_partition"):
                        dynamic_partition = line.split(": ")[-1] == "True"

                    if line.startswith("enhance"):
                        enhance = line.split(": ")[-1] == "True"

            if dynamic_partition and enhance:
                label = "EDP-SGD"
            elif dynamic_partition:
                label = "DP-SGD"
            else:
                label = "Sync-SGD"

            model_name = experiment.split("_")[-1]

            files = glob.glob(f"{experiment}/*.json")
            files.sort()

            for file in files:
                worker_type = file.split("_")[-1].split(".")[0]

                with open(file) as jsonfile:
                    json_data = json.load(jsonfile)

                batch_avg_time = json_data["batch"]["average_duration"]
                process_times = json_data["mean_batch_process_times"]

                from scipy.stats import gaussian_kde

                data = process_times
                density = gaussian_kde(data)

                xs = np.linspace(0, 1, 200)
                # density.covariance_factor = lambda: .25
                # density._compute_covariance()
                plt.hist(data, 100, label=f"{label} - Worker {worker_type}")
                # plt.title(f"{model_name}")

                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                plt.legend()
                plt.xlabel("Batch Process Time (sec)")
                plt.ylabel("Frequency")
                plt.savefig(f"./plots/process_times_histogram_{model_name}_{dynamic_partition}_{reducer}.svg")
            plt.show()


if __name__ == "__main__":
    root_log_path = "./logs/plot_logs/"

    plot_loss_curves(os.path.join(root_log_path, "convergence"))
    plot_loss_time_curves(os.path.join(root_log_path, "convergence"))
    plot_top1_accuracy_curves(os.path.join(root_log_path, "convergence"))
    plot_top1_accuracy_time_curves(os.path.join(root_log_path, "convergence"))
    plot_top5_accuracy_curves(os.path.join(root_log_path, "convergence"))

    plot_process_times_histogram(os.path.join(root_log_path, "process_times"))
