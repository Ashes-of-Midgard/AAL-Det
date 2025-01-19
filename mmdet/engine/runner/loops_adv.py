from typing import Sequence
import os
import os.path as osp

import torch
import torch.nn.functional as F
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
        init_adv_noises = [torch.randn_like(x.to(torch.float), device=data_batch['inputs'][0].device, dtype=torch.float, requires_grad=True) for x in data_batch['inputs']]
        # 2. Perform a clean forward propagation
        init_adv_inputs = [data_batch['inputs'][i]+0.01*init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        init_adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        self.runner.model.train_step(
            init_adv_data_batch, optim_wrapper=self.runner.optim_wrapper)
        # 3. Perform an adversarial training
        adv_noises = [init_adv_noises[i]+torch.sign(init_adv_noises[i].grad).detach() for i in range(len(init_adv_noises))]
        adv_inputs = [data_batch['inputs'][i]+0.01*adv_noises[i] for i in range(len(init_adv_noises))]
        adv_data_batch = {
            'inputs': adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        outputs = self.runner.model.train_step(
            adv_data_batch, optim_wrapper=self.runner.optim_wrapper)

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
        init_adv_noises = [torch.randn_like(x.to(torch.float), device=data_batch['inputs'][0].device, requires_grad=True) for x in data_batch['inputs']]
        # 2. Perform a clean forward propagation
        init_adv_inputs = [data_batch['inputs'][i]+0.01*init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        init_adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        _, attns = self.runner.model.train_step(
            init_adv_data_batch, optim_wrapper=self.runner.optim_wrapper)
        # 3. Perform an adversarial training
        adv_noises = [init_adv_noises[i]+torch.sign(init_adv_noises[i].grad).detach() for i in range(len(init_adv_noises))]
        adv_inputs = [data_batch['inputs'][i]+0.01*attns[i].to(adv_noises[i].device)*adv_noises[i] for i in range(len(init_adv_noises))]
        adv_data_batch = {
            'inputs': adv_inputs,
            'data_samples': data_batch['data_samples']
        }
        outputs,_ = self.runner.model.train_step(
            adv_data_batch, optim_wrapper=self.runner.optim_wrapper)

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
    
    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        # self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        
        # Adopting FGSM attack
        # 1. Initialize adversarial noises
        init_adv_noises = [torch.randn_like(x.to(torch.float), device=data_batch['inputs'][0].device, dtype=torch.float, requires_grad=True) for x in data_batch['inputs']]
        # 2. Perform a clean forward propagation
        init_adv_inputs = [data_batch['inputs'][i]+0.01*init_adv_noises[i] for i in range(len(data_batch['inputs']))]
        init_adv_data_batch = {
            'inputs': init_adv_inputs,
            'data_samples': data_batch['data_samples']
        }

        self.runner.model.train()
        if is_model_wrapper(self.runner.model):
            init_adv_data_batch = self.runner.model.module.data_preprocessor(init_adv_data_batch, True)
        else:
            init_adv_data_batch = self.runner.model.data_preprocessor(init_adv_data_batch, True)

        try: # Don't switch the order of try-except
            losses, attns = self.runner.model._run_forward(init_adv_data_batch, mode='loss')
        except:
            losses = self.runner.model._run_forward(init_adv_data_batch, mode='loss')
        # 3. Backpropagate the loss to generate adversarial noises
        loss_all = []
        for loss_values in losses.values():
            if isinstance(loss_values, torch.Tensor):
                loss_all.append(loss_values)
            elif isinstance(loss_values, list):
                loss_all.extend(loss_values)
        loss_sum = sum(loss_all)
        loss_sum.backward()
        adv_noises = [init_adv_noises[i]+torch.sign(init_adv_noises[i].grad).detach() for i in range(len(init_adv_noises))]
        
        self.runner.model.eval()
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
                try: # if model outputs attns
                    attns = torch.stack([F.interpolate(attn, init_adv_data_batch['inputs'].shape[2:4]) for attn in attns]).mean(dim=0)
                    attns = [attns[i, :, 0:data_batch['inputs'][i].shape[1], 0:data_batch['inputs'][i][2]].detach() for i in range(len(attns))]
                except:
                    attns = None
                for i in range(len(adv_noises)):
                    ori_shape = data_batch['data_samples'][i].ori_shape
                    ori_shape = (ori_shape[1], ori_shape[0])
                    name = data_batch['data_samples'][i].img_path.split('/')[-1].split('.png')[0]
                    tensor_to_pil_image(adv_noises[i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_adv_noise.png'))
                    tensor_to_pil_image(data_batch['inputs'][i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_ori.png'))
                    tensor_to_pil_image(adv_data_batch['inputs'][i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_adv.png'))
                    if attns is not None:
                        tensor_to_pil_image(attns[i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_attn_map.png'))
                        tensor_to_pil_image(attns[i]*adv_noises[i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_selective_adv_noise.png'))
                        tensor_to_pil_image(data_batch['inputs'][i]+0.01*attns[i]*adv_noises[i]).resize(ori_shape).save(osp.join(self.runner._log_dir, self.vis_dir,f'{name}_selective_adv.png'))

            self.runner.call_hook(
                'after_test_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)