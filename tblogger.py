import atexit
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter


class TBLogger(object):
    """
    xyz_dummies: stretch the screen with empty plots so the legend would
                 always fit for other plots
    """
    def __init__(self, local_rank, log_dir, interval=1, rescale_time_axis=1,
                 dummies=False):
        self.enabled = (local_rank == 0)
        self.interval = interval
        self.rescale_time_axis = rescale_time_axis
        self.cache = {}
        if local_rank == 0:
            self.summary_writer = SummaryWriter(
                log_dir=log_dir, flush_secs=120, max_queue=200)
            atexit.register(self.summary_writer.close)
            if dummies:
                for key in ('aaa', 'zzz'):
                    self.summary_writer.add_scalar(key, 0.0, 1)

    def log_value(self, step, key, val, stat='mean'):
        if self.enabled:
            if key not in self.cache:
                self.cache[key] = []
            self.cache[key].append(val)
            if len(self.cache[key]) == self.interval:
                agg_val = getattr(np, stat)(self.cache[key])
                self.summary_writer.add_scalar(key, agg_val,
                                               step * self.rescale_time_axis)
                del self.cache[key]

    def log_meta(self, step, meta):
        for k, v in meta.items():
            self.log_value(step, k, v)

    def log_grads(self, step, model):
        if self.enabled:
            norms = [p.grad.norm().item() for p in model.parameters()
                     if p.grad is not None]
            for stat in ('max', 'min', 'mean'):
                self.log_value(step, f'grad_{stat}', getattr(np, stat)(norms),
                               stat=stat)


class Logger(TBLogger):
    def log_meta(self, epoch, step, key, meta):
        meta = {mkey: v.item() if torch.is_tensor(v) else v
                for mkey, v in meta.items()}

        if False:  # stdout
            msg = f'{key} {epoch}:{step:>d}'
            for mkey,v in meta.items():
                msg += f' {mkey}={v:>14.9f}'
            print(msg)

        tb_meta = {f'{key}_{mkey}': v for mkey, v in meta.items()}
        super(Logger, self).log_meta(step, tb_meta)
