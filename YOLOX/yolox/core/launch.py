#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yolox.utils.dist as comm
from yolox.utils import configure_nccl

import os
import subprocess
import sys
import time

__all__ = ["launch"]


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            dist_url = "{}:{}".format(
                os.environ.get("MASTER_ADDR", None),
                os.environ.get("MASTER_PORT", "None"),
            )
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            _distributed_worker(
                local_rank,
                main_func,
                world_size,
                num_gpus_per_machine,
                num_machines,
                machine_rank,
                backend,
                dist_url,
                args,
            )
            exit()
        launch_by_subprocess(
            sys.argv,
            world_size,
            num_machines,
            machine_rank,
            num_gpus_per_machine,
            dist_url,
            args,
        )
    else:
        main_func(*args)


def launch_by_subprocess(
    raw_argv,
    world_size,
    num_machines,
    machine_rank,
    num_gpus_per_machine,
    dist_url,
    args,
):
    assert (
        world_size > 1
    ), "subprocess mode doesn't support single GPU, use spawn mode instead"

    if dist_url is None:
        # ------------------------hack for multi-machine training -------------------- #
        if num_machines > 1:
            master_ip = subprocess.check_output(["hostname", "--fqdn"]).decode("utf-8")
            master_ip = str(master_ip).strip()
            dist_url = "tcp://{}".format(master_ip)
            ip_add_file = "./" + args[1].experiment_name + "_ip_add.txt"
            if machine_rank == 0:
                port = _find_free_port()
                with open(ip_add_file, "w") as ip_add:
                    ip_add.write(dist_url+'\n')
                    ip_add.write(str(port))
            else:
                while not os.path.exists(ip_add_file):
                    time.sleep(0.5)

                with open(ip_add_file, "r") as ip_add:
                    dist_url = ip_add.readline().strip()
                    port = ip_add.readline()
        else:
            dist_url = "tcp://127.0.0.1"
            port = _find_free_port()

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = dist_url
    current_env["MASTER_PORT"] = str(port)
    current_env["WORLD_SIZE"] = str(world_size)
    assert num_gpus_per_machine <= torch.cuda.device_count()

    if "OMP_NUM_THREADS" not in os.environ and num_gpus_per_machine > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        logger.info(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(
                current_env["OMP_NUM_THREADS"]
            )
        )

    processes = []
    for local_rank in range(0, num_gpus_per_machine):
        # each process's rank
        dist_rank = machine_rank * num_gpus_per_machine + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = ["python3", *raw_argv]

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    num_machines,
    machine_rank,
    backend,
    dist_url,
    args,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    configure_nccl()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    if global_rank == 0 and os.path.exists(
        "./" + args[1].experiment_name + "_ip_add.txt"
    ):
        os.remove("./" + args[1].experiment_name + "_ip_add.txt")

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    args[1].local_rank = local_rank
    args[1].num_machines = num_machines

    # Setup the local process group (which contains ranks within the same machine)
    # assert comm._LOCAL_PROCESS_GROUP is None
    # num_machines = world_size // num_gpus_per_machine
    # for i in range(num_machines):
    # ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
    # pg = dist.new_group(ranks_on_i)
    # if i == machine_rank:
    # comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)
