from typing import Sequence
import os
import os.path as osp

import torch
from PIL import Image

from mmengine.model import is_model_wrapper
from mmengine.runner import EpochBasedTrainLoop, TestLoop
from mmengine.runner.loops import _update_losses, _parse_losses
from mmengine.runner.amp import autocast
from mmengine.dataset.base_dataset import Compose

from mmdet.registry import LOOPS


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Convert Tensor to PIL image type
    """
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
    tensor = (tensor * 255).type(torch.uint8)  # Convert to [0,255] uint8 value
    tensor = tensor.permute(1, 2, 0)  # convert [3, H, W] order to [H, W, 3]
    np_array = tensor.cpu().numpy()  # convert to numpy array
    pil_image = Image.fromarray(np_array)  # convert to PIL image
    return pil_image


# def de_normalize(tensor: torch.Tensor, mean, std) -> torch.Tensor:
#     de_normalized_tensor = tensor
#     de_normalized_tensor[0] = (tensor[0] * std[0]) + mean[0]
#     de_normalized_tensor[1] = (tensor[1] * std[1]) + mean[1]
#     de_normalized_tensor[2] = (tensor[2] * std[2]) + mean[2]
#     return de_normalized_tensor


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

        # Adopting FGSM attack
        # 1. Initialize all zero adversarial noises
        init_adv_noises = [torch.randn_like(x, device=data_batch['inputs'][0].device, dtype=torch.float, requires_grad=True) for x in data_batch['inputs']]
        # 2. Perform a clean forward propagation
        init_adv_inputs = [data_batch['inputs'][i]+0.01*init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        self.runner.model.train_step(
            adv_data_batch, optim_wrapper=self.runner.optim_wrapper)
        # 3. Perform an adversarial training
        adv_noises = [init_adv_noises[i]+torch.sign(init_adv_noises[i].grad).detach().requires_grad_(False) for i in range(len(init_adv_noises))]
        adv_inputs = [data_batch['inputs'][i]+0.01*adv_noises[i] for i in range(len(init_adv_noises))]
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


@LOOPS.register_module()
class AALTrainLoop(EpochBasedTrainLoop):
    """
        Customized associative adversarial training loop
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

        # Adopting FGSM attack
        # 1. Initialize random adversarial noises
        init_adv_noises = [torch.randn_like(x, device=data_batch['inputs'][0].device, requires_grad=True) for x in data_batch['inputs']]
        # 2. Perform a clean forward propagation
        init_adv_inputs = [data_batch['inputs'][i]+0.01*init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        self.runner.model.train_step(
            adv_data_batch, optim_wrapper=self.runner.optim_wrapper)
        # Distributed training is not tested. No guarantee.
        if is_model_wrapper(self.runner.model):
            attn = self.runner.model.module.attn
        else:
            attn = self.runner.model.attn.to(data_batch['inputs'][0].device)
        # 3. Perform an adversarial training
        adv_noises = [init_adv_noises[i]+torch.sign(init_adv_noises[i].grad).detach().requires_grad_(False) for i in range(len(init_adv_noises))]
        adv_inputs = [data_batch['inputs'][i]+0.01*attn[i]*adv_noises[i] for i in range(len(init_adv_noises))]
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


@LOOPS.register_module()
class AdvTestLoop(TestLoop):
    """
        Customized adversarial testing loop
    """
    def __init__(self, runner, dataloader, evaluator, fp16 = False, vis_dir=None):
        super().__init__(runner, dataloader, evaluator, fp16)
        self.vis_dir = vis_dir
        if vis_dir is not None:
            self.num_tsne_samples = 100
            self.tsne_data = {
                'target_embedding': [],
                'background_embedding': [],
                'target_embedding_adv': [],
                'background_embedding_adv': []
            }

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        
        # Adopting FGSM attack
        # 1. Initialize all zero adversarial noises
        init_adv_noises = [torch.randn_like(x, device=data_batch['inputs'][0].device, dtype=torch.float, requires_grad=True) for x in data_batch['inputs']]
        # 2. Perform a clean forward propagation
        init_adv_inputs = [data_batch['inputs'][i].to(torch.float).requires_grad_(True)+0.01*init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        adv_data_batch = self.runner.model.data_preprocessor(adv_data_batch, True)
        losses = self.runner.model._run_forward(adv_data_batch, mode='loss')
        # 3. Backpropagate the loss to generate adversarial noises
        loss_all = []
        for loss_values in losses.values():
            if isinstance(loss_values, torch.Tensor):
                loss_all.append(loss_values)
            elif isinstance(loss_values, list):
                loss_all.extend(loss_values)
        loss_sum = sum(loss_all)
        loss_sum.backward()
        adv_noises = [init_adv_noises[i]+torch.sign(init_adv_noises[i].grad).detach().requires_grad_(False) for i in range(len(init_adv_noises))]
        with torch.no_grad():
            adv_inputs = [data_batch['inputs'][i]+0.01*adv_noises[i] for i in range(len(init_adv_noises))]
            adv_data_batch = {
                'inputs': adv_inputs,
                'data_samples': data_batch['data_samples']
            }

            with autocast(enabled=self.fp16):
                outputs = self.runner.model.test_step(adv_data_batch)

            outputs, self.test_loss = _update_losses(outputs, self.test_loss)

            self.evaluator.process(data_samples=outputs, data_batch=adv_data_batch)

            # Draw visualization result
            if self.vis_dir is not None:
                os.makedirs(osp.join(self.runner._log_dir, self.vis_dir), exist_ok=True)
                for i in range(len(adv_noises)):
                    ori_shape = data_batch['data_samples'][i].ori_shape
                    ori_shape = (ori_shape[1], ori_shape[0])
                    name = data_batch['data_samples'][i].img_path.split('/')[-1].split('.png')[0]
                    tensor_to_pil_image(adv_noises[i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_adv_noise.png'))
                    tensor_to_pil_image(data_batch['inputs'][i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_ori.png'))
                    tensor_to_pil_image(data_batch['inputs'][i]+0.01*adv_noises[i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_adv.png'))

            self.runner.call_hook(
                'after_test_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)