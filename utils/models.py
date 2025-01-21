import torchvision.transforms
from robustbench.utils import load_model
from global_settings import undefended_model_names, cifar10_checkpoint_dir
from pytorch_cifar.main import model_dict, models_args
import torch
import os
from torchvision.models import get_model, get_model_weights
from torchvision import transforms
import torch.nn as nn

import torch.nn.functional as F


class DiversityWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.resize_rate = 0.9
        self.diversity_prob = 0.5
        self.model = model

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, x):
        self.model(self.input_diversity(x))


class InceptionV3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.transform = torchvision.transforms.Resize((299, 299))

    def forward(self, x):
        # change the input size to the correct one
        # return self.model(self.transform(x)).logits
        x = self.model(self.transform(x))
        if hasattr(x, "logits"):
            return x.logits
        else:
            return x


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        out = None
        for model in self.models:
            if out is None:
                out = model(x)
            else:
                out += model(x)
        return out


class UndefendedWrapper(torch.nn.Module):
    def __init__(self, model, model_name, dataset: str):
        super().__init__()
        self.model = model
        if dataset == "cifar10":
            self.transform = transforms.Compose([
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            weight = get_model_weights(model_name).DEFAULT
            self.transform = weight.transforms(antialias=True)  # same setting as pytorch

    def forward(self, x):
        return self.model(self.transform(x))


def load_pytorch_model(model_name, dataset_name):
    if dataset_name == "imagenet":
        model = get_model(model_name, weights="DEFAULT")
        if model_name == "inception_v3":
            # expects a different input size, so we wrap it around a size changing wrapper
            model = InceptionV3Wrapper(model)
    else:
        checkpoint = torch.load(os.path.join(cifar10_checkpoint_dir, f"{model_name}_ckpt.pt"))
        model = model_dict.get(model_name, None)
        if model is None:
            raise NotImplementedError(f"Only {list(model_dict.keys())}")
        else:
            if models_args.get(model_name, None) is not None:
                model = model(**models_args[model_name])
            else:
                model = model()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint["net"])
        model = model.module
    model.eval()
    model = UndefendedWrapper(model, model_name, dataset_name).eval()
    return model


def load_defended_or_undefended(model_name, dataset_name, threat_model, model_dir):
    if model_name == "Chen2024Data_WRN_50_2":
        from utils.robustbench_utils import load_model as fixed_load_model
        return fixed_load_model(model_name=model_name, dataset=dataset_name, threat_model=threat_model,
                                model_dir=model_dir)
    if model_name in undefended_model_names[dataset_name]:
        return load_pytorch_model(model_name, dataset_name)
    else:
        return load_model(model_name=model_name, dataset=dataset_name, threat_model=threat_model,
                          model_dir=model_dir)


