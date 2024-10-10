## training loop for fm:


import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
import math
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from utils_training import ema, infiniteloop, warmup_lr
from tqdm import trange


# training loop begins


def training_loop(
    run_dir=".",
    dataset_kwargs={},
    data_loader_kwargs={},
    network_kwargs={},
    trajectory_kwargs={},
    optimizer_kwargs={},
    augment_kwargs=None,
    seed=0,
    warmup=45000,
    batch_size=256,
    batch_gpu=None,  # limit batch size per gpu, None = no limit
    total_steps=4000000,  # lipman et al. 391k
    lr=5e-4,
    ema_decay=0.9999,
    loss_scaling=1,
    snapshot_steps=40000,
    resume_pkl=None,
    resume_state_dump=None,
    cudnn_benchmark=True,
    device=torch.device("cuda"),
):
    # Initialize
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch_size per gpu:
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == num_accumulation_rounds * batch_gpu * dist.get_world_size()

    # Load dataset
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_word_size(),
        seed=seed,
    )

    dataset_iterator = iter(
        torch.utils.data.Dataloader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )

    # Construct network

    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_resolution=dataset_obj.resolution,
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    v_net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)

    v_net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros(
                [batch_gpu, v_net.img_resolution, v_net.img_resolution],
                device=device,
            )
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, v_net.label_dim], device=device)
            misc.print_module_summary(v_net, [images, sigma, labels], max_nesting=2)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    dist.print0("Model params: %.2f M" % (model_size / 1024 / 1024))

    # setup flow trajectory
    dist.print0("setting up flow trajectory...")

    trajectory = dnnlib.util.construct_class_by_name(
        **trajectory_kwargs
    )  # training.trajectory.(OT|VPD|VED|subVPD)FM

    # setup optimizer and scheduler
    dist.print0("setting up optimizer...")
    optimizer = dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )

    # distributed training setup:
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    ema = copy.deepcopy(net).eval().require_grads_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        misc.copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        misc.copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train

    dist.print0(f"Training for total_steps: {total_steps} and batch_size: {batch_size}")

    # calculate the total epochs
    steps_per_epoch = math.ceil(len(dataset_obj) / batch_size)
    num_epochs = math.ceil(total_steps / steps_per_epoch)

    with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for step in step_pbar:
                    global_step += step
                    optimizer.zero_grad(set_to_none=True)
                    # accumulate gradient
                    for round_idx in range(num_accumulation_rounds):
                        with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    
                        x1, labels = next(dataset_iterator)
                        x1 = x1.to(device).to(torch.float32) / 127.5 - 1
                        labels = labels.to(device)
                        
                        loss = trajectory.loss(v_net,x1)
                        training_stats.report("Loss/loss", loss)
                        loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                    
                    # update weights:

                    for param in net.parameters():
                        if param.grad is not None:
                            torch.nan_to_num(
                                param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                            )

                    optimizer.step()  
                    scheduler.step()                 
                    
                    
                    ema(net_model, ema_model, ema_decay)  # new
                    
                    # Save network snapshot.
                    if (snapshot_steps is not None) and global_step % snapshot_steps == 0:
                        data = dict(
                            ema=ema,
                            trajectory=trajectory,
                            augment_pipe=augment_pipe,
                            dataset_kwargs=dict(dataset_kwargs),
                        )
                        for key, value in data.items():
                            if isinstance(value, torch.nn.Module):
                                value = copy.deepcopy(value).eval().requires_grad_(False)
                                misc.check_ddp_consistency(value)
                                data[key] = value.cpu()
                            del value  # conserve memory
                        if dist.get_rank() == 0:
                            with open(
                                os.path.join(run_dir, f"network-snapshot-{global_step:06d}.pkl"),
                                "wb",
                            ) as f:
                                pickle.dump(data, f)
                        del data  # conserve memory


                    # Update logs.
                    training_stats.default_collector.update()
                    if dist.get_rank() == 0:
                        if stats_jsonl is None:
                            stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
                        stats_jsonl.write(
                            json.dumps(
                                dict(
                                    training_stats.default_collector.as_dict(),
                                    timestamp=time.time(),
                                )
                            )
                            + "\n"
                        )
                        stats_jsonl.flush()


    # Done.
    dist.print0()
    dist.print0("Exiting...")

