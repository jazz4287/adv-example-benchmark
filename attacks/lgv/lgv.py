import torchattacks

from attacks.attack import Attack
from global_settings import epsilon, device, lgv_save_path
from utils.data import load_data
import os
import torch
from copy import deepcopy
from attacks.lgv.torchattack_lgv import LGV



class LGVAttack(Attack):
    variants = ["A", "B"]
    combinations = {"A": ["di", "ti", "di-ti"]}
    def __init__(self, variant, threat_model, dataset_name, di: bool = False, ti: bool = False, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        assert variant is not None, (f"Please specify a variant for the LGV attack. Either (A) for the transferable"
                                     f" version or (B) for the white-box version")
        if self.variant == "A":
            self.has_surrogate = True
        self.attack = None
        self.model_name = None
        self.di = di
        self.ti = ti
        # paper uses a batch size of 256

    def set_model(self, model, model_name):
        model = model.to(device)
        self.model_name = model_name
        if model_name == "Peng2023Robust" and self.variant == "B":
            if self.dataset_name == "cifar10":
                batch_size = 240  # can't fit all in memory unfortunately
            else:
                batch_size = 128
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "Singh2023Revisiting_ConvNeXt-L-ConvStem" and self.dataset_name == "imagenet"
              and self.variant == "B"):
            batch_size = 192  # can't fit all in memory unfortunately
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "Liu2023Comprehensive_ConvNeXt-L") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 192
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "convnext_large") and (self.dataset_name == "imagenet") and (self.variant == "B"):
            batch_size = 192
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "Liu2023Comprehensive_Swin-L") and (self.dataset_name == "imagenet"):
            batch_size = 192
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_b4") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 96
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_b5") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 48
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_b6") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 32
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_b7") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 16
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_v2_l") and (self.dataset_name == "image22net") and (
                self.variant == "B"):
            batch_size = 32
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_v2_m") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 64
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "efficientnet_v2_s") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 128
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "regnet_y_128gf") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 32
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "swin_v2_b") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 192
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "vit_h_14") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 16
            print(f"Reducing batch size to {batch_size}")
        elif (model_name == "vit_l_16") and (self.dataset_name == "imagenet") and (
                self.variant == "B"):
            batch_size = 192
            print(f"Reducing batch size to {batch_size}")
        else:
            batch_size = 256
        trainloader = load_data(dataset_name=self.dataset_name, threat_model=self.threat_model, model_name=model_name,
                                train=True, batch_size=batch_size, use_robust_5000=False)
        eps = epsilon[self.dataset_name]
        if self.dataset_name == "imagenet":
            if self.variant == "B":
                num_epochs = 1
            else:
                num_epochs = 5  # we can afford more cuz we only do it once
        else:
            num_epochs = 10
        if not self.di and not self.ti:
            base_attack = torchattacks.BIM
        elif self.di and not self.ti:
            base_attack = torchattacks.DIFGSM
        elif self.di and self.ti:
            base_attack = torchattacks.TIFGSM  # TIFGSM implements DI by default
        else:
            raise NotImplementedError
        self.attack = LGV(model, trainloader, epochs=num_epochs,
                                       eps=eps, alpha=4 / 255 / 10, steps=50, attack_class=base_attack)

        self.attack.device = device
        if model_name == 'Bai2023Improving_edm' or model_name == 'Peng2023Robust': #or model_name == "Singh2023Revisiting_ConvNeXt-L-ConvStem":
            self.attack.use_cpu_offload = True
        save_path = os.path.join(lgv_save_path, self.dataset_name, model_name)
        # check if the last model in the list exists ( num_epochs*4 models (4 is the num of models per epoch))
        if os.path.exists(os.path.join(save_path,
                                       f"lgv_model_{(num_epochs*4)-1:05}.pt")):
            model_list = []
            i = 0
            while os.path.exists(os.path.join(save_path, f"lgv_model_{i:05}.pt")):
                checkpoint = torch.load(os.path.join(save_path, f"lgv_model_{i:05}.pt"))
                model_copy = deepcopy(model).to(torch.device("cpu"))
                model_copy.load_state_dict(checkpoint["state_dict"])
                model_list.append(model_copy.to(torch.device("cpu")))
                i += 1
            self.attack.load_models(model_list)
        else:
            self.attack.collect_models()
            self.attack.save_models(save_path)

    def forward(self, x, y):
        if self.attack is None:
            print("Please set the model before attacking")
            raise AttributeError
        return self.attack(x, y)

    def base_name(self):
        return "LGV"

    def name(self):
        name = ""
        if self.di:
            name += "DI-"
        if self.ti:
            name += "TI-"
        name += "LGV"

        return name
