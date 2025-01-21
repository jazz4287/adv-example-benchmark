import torch
import random
import numpy as np
from global_settings import not_working_models


def get_not_working_model_list():
    not_working_model_lists = {}
    for dataset in not_working_models.keys():
        not_working_model_lists[dataset] = []
        for attack in not_working_models[dataset]:
            for model in not_working_models[dataset][attack]:
                if model not in not_working_model_lists[dataset]:
                    not_working_model_lists[dataset].append(model)

    return not_working_model_lists


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def move_model_to_device(model, device_map):
    for name, param in model.named_parameters():
        if name in device_map:
            device = device_map[name]
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)
    for name, buffer in model.named_buffers():
        if name in device_map:
            device = device_map[name]
            buffer.data = buffer.data.to(device)


def set_seeds(seed_num: int = 154830):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    print(f"seed set to {seed_num}")