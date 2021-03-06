import os
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist

from seed import set_seed
from timer import Timer
from logger import Logger
from metrics import AverageMeter
from model_dispatcher import CIFAR
from reducer import NoneAllReducer


config = dict(
    runs=5,
    distributed_backend="nccl",
    num_epochs=150,
    batch_size=128 * 2,
    architecture="ResNet50",
    # architecture="VGG16",
    local_steps=1,
    delay_type="constant",
    delay_constant=0.1,
    # delay_type="gamma",
    # delay_type="exponential",
    # scale_factor=0.25,
    dynamic_partition=True,
    # dynamic_partition = False,
    enhance=True,
    # enhance=False,
    seed=42,
    log_verbosity=2,
    lr=0.1,
)


def initiate_distributed():
    env_dict = {key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}

    print(f"[{os.getpid()}] Initializing Process Group with: {env_dict}")
    dist.init_process_group(backend=config["distributed_backend"], init_method="env://")

    print(
        f"[{os.getpid()}] Initialized Process Group with: RANK = {dist.get_rank()}, "
        + f"WORLD_SIZE = {dist.get_world_size()}"
        + f", backend={dist.get_backend()}"
    )


def train(local_rank, world_size):
    logger = Logger(config, local_rank)
    best_accuracy = {"top1": [0] * config["runs"], "top5": [0] * config["runs"]}

    for run in range(config["runs"]):
        set_seed(config["seed"])
        device = torch.device(f"cuda:{local_rank}")
        timer = Timer(verbosity_level=config["log_verbosity"])

        reducer = NoneAllReducer(device, timer)
        lr = config["lr"]
        bits_communicated = 0

        global_iteration_count = 0
        model = CIFAR(device, timer, config["architecture"], config["seed"] + local_rank)

        send_buffers = [torch.zeros_like(param) for param in model.parameters]
        partitions = [1 / world_size] * world_size

        # optimizer = optim.SGD(params=model.parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD(params=model.parameters, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=0)

        for epoch in range(config["num_epochs"]):
            if local_rank == 0:
                logger.log_info(
                    "epoch info",
                    {"Progress": epoch / config["num_epochs"], "Current_epoch": epoch},
                    {"lr": scheduler.get_last_lr()},
                )

            logger.log_info(
                "partition_info",
                {"Current_epoch": epoch},
                {"Rank": local_rank, "Current_size": partitions[local_rank]},
            )

            epoch_metrics = AverageMeter(device)

            train_loader = model.train_dataloader(partitions, config["batch_size"])
            for i, batch in enumerate(train_loader):
                global_iteration_count += 1
                epoch_frac = epoch + i / model.len_train_loader

                with timer("batch", epoch_frac):
                    with timer("batch.process", epoch_frac):
                        _, grads, metrics = model.batch_loss_with_gradients(batch)
                        epoch_metrics.add(metrics)

                        if local_rank == 1:
                            import time

                            if config["delay_type"] == "constant":
                                time.sleep(config["delay_constant"])
                            elif config["delay_type"] == "gamma":
                                time.sleep((np.random.gamma(config["scale_factor"])))
                            elif config["delay_type"] == "exponential":
                                time.sleep((np.random.exponential(config["scale_factor"])))

                    if global_iteration_count % config["local_steps"] == 0:
                        with timer("batch.accumulate", epoch_frac, verbosity=2):
                            for grad, send_buffer in zip(grads, send_buffers):
                                if config["enhance"]:
                                    send_buffer[:] = grad * partitions[local_rank]
                                else:
                                    send_buffer[:] = grad

                        with timer("batch.reduce", epoch_frac):
                            bits_communicated += reducer.reduce(send_buffers, grads)

                    with timer("batch.step", epoch_frac, verbosity=2):
                        optimizer.step()

            scheduler.step()

            mean_batch_process_time = torch.tensor(
                sum(timer.batch_process_times) / len(timer.batch_process_times), device=device, dtype=torch.float32
            )
            timer.mean_batch_process_times.append(mean_batch_process_time.item())
            print(f"Mean Batch Time for Rank {local_rank}:", mean_batch_process_time.item())

            timer.batch_process_times = []
            collected_batch_process_times = [torch.empty_like(mean_batch_process_time) for _ in range(world_size)]

            if world_size > 1:
                batch_process_time_workers_op = torch.distributed.all_gather(
                    tensor_list=collected_batch_process_times, tensor=mean_batch_process_time, async_op=True
                )
                batch_process_time_workers_op.wait()
            else:
                collected_batch_process_times = [mean_batch_process_time]

            if config["dynamic_partition"]:
                partitions = [
                    (partition / batch_process_time).item()
                    for batch_process_time, partition in zip(collected_batch_process_times, partitions)
                ]
                partitions = [partition / sum(partitions) for partition in partitions]

            with timer("epoch_metrics.collect", epoch, verbosity=2):
                epoch_metrics.reduce()
                if local_rank == 0:
                    for key, value in epoch_metrics.values().items():
                        logger.log_info(
                            key,
                            {"value": value, "epoch": epoch, "bits": bits_communicated},
                            tags={"split": "train"},
                        )

            with timer("test.last", epoch):
                test_stats = model.test(partitions)
                if local_rank == 0:
                    for key, value in test_stats.values().items():
                        logger.log_info(
                            key,
                            {"value": value, "epoch": epoch, "bits": bits_communicated},
                            tags={"split": "test"},
                        )

                        if "top1_accuracy" == key and value > best_accuracy["top1"][run]:
                            best_accuracy["top1"][run] = value
                            logger.save_model(model)

                        if "top5_accuracy" == key and value > best_accuracy["top5"][run]:
                            best_accuracy["top5"][run] = value

            if local_rank == 0:
                logger.epoch_update(run, epoch, epoch_metrics, test_stats)

        if local_rank == 0:
            print(timer.summary())

    logger.summary_writer(timer, best_accuracy, bits_communicated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    local_rank = args.local_rank

    initiate_distributed()
    world_size = dist.get_world_size()
    train(local_rank, world_size)
