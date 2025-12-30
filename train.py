import argparse
from argparse import Namespace
from pathlib import Path
import warnings
import os
import random

import torch
import pytorch_lightning as pl
import yaml
import numpy as np

from lightning_modules import LigandPocketDDPM


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    os.makedirs('logs/', exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    p.add_argument('--early_stop_patience', type=int, default=10, help='Patience (in epochs) for early stopping')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    # ---------------------- Seed fixation ----------------------
    seed = int(args.seed) if getattr(args, 'seed', None) is not None else 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    pl.seed_everything(seed, workers=True)
    # -----------------------------------------------------------

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()

    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes,
        alpha_param=args.alpha_param,
        alpha_power=args.alpha_power,
        # guidance_scale=args.guidance_scale,
        p_unconditioned=args.p_unconditioned,
    )

    logger = pl.loggers.MLFlowLogger(
        save_dir=args.logdir,
        experiment_name='DiffInt_training',
        run_name=args.run_name,
        tags={'dataset': args.dataset},
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    callbacks = [checkpoint_callback]
    if getattr(args, 'early_stopping', False):
        early_stop_cb = pl.callbacks.EarlyStopping(
            monitor='loss/val',
            patience=args.early_stop_patience,
            mode='min',
            verbose=True,
        )
        callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator='gpu', devices=args.gpus,
        strategy='ddp'
        #strategy=('ddp' if args.gpus > 1 else None)
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
