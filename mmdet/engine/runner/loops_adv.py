from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop, EpochBasedTrainLoop

from mmdet.registry import LOOPS


@LOOPS.register_module()
class AdvTrainLoop(EpochBasedTrainLoop):
    """
        Customized adversarial training loop
    """
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        init_adv_noises = [torch.zeros_like(x, device=data_batch['inputs'][0].device, requires_grad=True) for x in data_batch['inputs']]
        init_adv_inputs = [data_batch['inputs'][i]+init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        adv_data_batch = self.runner.model.data_preprocessor(adv_data_batch, True)
        losses = self.runner.model._run_forward(adv_data_batch, mode='loss')
        loss_all = []
        for loss_list in losses.values():
            for value in loss_list:
                loss_all.append(value)
        loss_sum = sum(loss_all)
        print(loss_sum.requires_grad)
        loss_sum.backward()
        adv_noises = [torch.sign(init_adv_noises[i].grad).detach() for i in range(len(init_adv_noises))]
        adv_inputs = [data_batch['inputs'][i]+adv_noises[i] for i in range(len(init_adv_noises))]
        adv_data_batch = {
            'inputs': adv_inputs,
            'data_samples': data_batch['data_samples']
        }

        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1