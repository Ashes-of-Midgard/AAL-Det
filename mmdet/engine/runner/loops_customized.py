import torch

from mmengine.model import is_model_wrapper
from mmengine.runner import EpochBasedTrainLoop

from mmdet.registry import LOOPS


@LOOPS.register_module()
class CustomizedTrainLoop(EpochBasedTrainLoop):
    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        # # 计算当前模型权重与预训练权重逐参数的平均差值
        # model_state_dict_pretrained = torch.load('../models/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth')['state_dict']
        # if is_model_wrapper(self.runner.model):
        #     model_state_dict = self.runner.model.module.state_dict()
        # else:
        #     model_state_dict = self.runner.model.state_dict()
        # diff_avg = 0.
        # num_params = 0
        # for key, value in model_state_dict.items():
        #     if not key in model_state_dict_pretrained:
        #         print(f'{key} is not in pretrained model')
        #     else:
        #         diff_avg += torch.abs(value - model_state_dict_pretrained[key].cuda()).sum() # 累积参数差值
        #         num_params_this_layer = 1
        #         for i in range(len(value.shape)):
        #             num_params_this_layer *= value.shape[i] # 累积参数数量
        #         num_params += num_params_this_layer
        # diff_avg = diff_avg / num_params # 求参数差值平均值
        # print(f'Before epoch [{self._epoch}]: {diff_avg.item():.4f}')
        
        self.runner.val_loop.run() # 训练前进行一次测试

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

        self.runner.val_loop.run() # 训练后进行一次测试

        # # 再次计算训练后模型权重与预训练权重的逐元素平均差值
        # if is_model_wrapper(self.runner.model):
        #     model_state_dict = self.runner.model.module.state_dict()
        # else:
        #     model_state_dict = self.runner.model.state_dict()
        # diff_avg = 0.
        # num_params = 0
        # for key, value in model_state_dict.items():
        #     if not key in model_state_dict_pretrained:
        #         print(f'{key} is not in pretrained model')
        #     else:
        #         diff_avg += torch.abs(value - model_state_dict_pretrained[key].cuda()).sum() # 累积参数差值
        #         num_params_this_layer = 1
        #         for i in range(len(value.shape)):
        #             num_params_this_layer *= value.shape[i] # 累积参数数量
        #         num_params += num_params_this_layer
        # diff_avg = diff_avg / num_params # 求参数差值平均值
        # print(f'After epoch [{self._epoch}]: {diff_avg.item():.4f}')

        self.runner.call_hook('after_train')
        return self.runner.model