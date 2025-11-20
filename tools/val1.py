# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model (no visualization)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file (pth)')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # optional: replace ceph backend if required
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    # merge cfg options passed from CLI (like load_from=...)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set launcher
    cfg.launcher = args.launcher

    # ensure work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # set checkpoint to load
    cfg.load_from = args.checkpoint

    # ---- Disable any visualization hooks to avoid visualization code running ----
    # If default_hooks exists and contains 'visualization', remove it
    if 'default_hooks' in cfg and cfg.default_hooks is not None:
        if 'visualization' in cfg.default_hooks:
            # delete visualization hook to avoid visualization backend reading files
            del cfg.default_hooks['visualization']

    # Also remove visualizer / vis_backends if present
    if 'visualizer' in cfg:
        try:
            del cfg.visualizer
        except Exception:
            pass
    if 'vis_backends' in cfg:
        try:
            del cfg.vis_backends
        except Exception:
            pass

    # For safety, clear training-only settings so Runner won't require optimizer/train loop
    # If you already set test_dataloader in config, leave it; ensure train_dataloader/optim_wrapper/param_scheduler absent
    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None

    # Print debug info so you can confirm in log what's being used
    print(f"=== Running test_no_vis.py ===")
    print(f"Config file: {args.config}")
    print(f"Checkpoint to load: {args.checkpoint}")
    print(f"Work dir: {cfg.work_dir}")
    if 'test_dataloader' in cfg:
        print("test_dataloader found in cfg.")
    else:
        print("WARNING: test_dataloader not found in cfg. Ensure cfg defines validation/test dataloader.")
    # build the runner from config (this will build model, dataloader and evaluators)
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Extra confirmation log: mmengine will also log checkpoint loading,
    # but we print model device/name here:
    try:
        print("Model summary:", runner.model.__class__.__name__)
    except Exception:
        pass

    # start testing (this runs the TestLoop: inference + evaluation)
    runner.test()


if __name__ == '__main__':
    main()
