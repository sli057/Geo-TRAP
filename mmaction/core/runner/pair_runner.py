import time
import warnings
from itertools import zip_longest

import torch
import mmcv
from mmcv.runner import EpochBasedRunner, RUNNERS
from mmcv.runner.utils import get_host_info



def grouper(iterable, n=2, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def compose_pair_batch(data_batch_pair):
    data_batch1, data_batch2 = data_batch_pair
    if data_batch2 is None:
        return None
    if data_batch1[0]['imgs'].size() != data_batch2[0]['imgs'].size():
        return None
    data_batch = dict()
    data_batch['imgs1'] = data_batch1[0]['imgs'].reshape((-1,) +  data_batch1[0]['imgs'].shape[2:])
    data_batch['label1'] = data_batch1[0]['label']
    data_batch['flow1'] = data_batch1[1]['imgs'].reshape((-1,) +  data_batch1[1]['imgs'].shape[2:])
    data_batch['imgs2'] = data_batch2[0]['imgs'].reshape((-1,) +  data_batch2[0]['imgs'].shape[2:])
    data_batch['label2'] = data_batch2[0]['label']
    data_batch['flow2'] = data_batch2[1]['imgs'].reshape((-1,) +  data_batch2[1]['imgs'].shape[2:])
    return data_batch

@RUNNERS.register_module()
class PairRunner(EpochBasedRunner):
    """Pair-Input Epoch-based Runner.

    This runner train models epoch by epoch. The input of the models is a pair of samples.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader) // 2 # PairInput
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        # print(type(self.model))
        # <class 'mmcv.parallel.data_parallel.MMDataParallel'>
        # self.model.train_step()
        # exit()
        for i, data_batch_pair in enumerate(grouper(self.data_loader, n=2)):
            data_batch = compose_pair_batch(data_batch_pair)  # PairInput
            if data_batch is None:
                break
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        print('Now validating...')
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch_pair in enumerate(grouper(self.data_loader, n=2)):
            data_batch = compose_pair_batch(data_batch_pair)  # PairInput
            if data_batch is None:
                break
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        # [ < torch.utils.data.dataloader.DataLoader object at 0x7f6fe72c9690 >]
        # [('train', 1)]
        # 45
        # {}
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow # epochs is not used afterward
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i]) // 2  # PairInput
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)

        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

