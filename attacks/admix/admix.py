from attacks.attack import Attack
from global_settings import device, epsilon
import torch
from attacks.admix.original_admix import Admix


class AdmixAttack(Attack):
    variants = ["A", "B"]
    combinations = {"A": ["di", "ti", "di-ti"]}
    def __init__(self, variant, threat_model, dataset_name, di: bool = False, ti: bool = False, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        self.attack = None
        self.model_name = None
        assert variant is not None, (f"Please specify a variant for the Admix attack. Either (A) for the transferable"
                                     f" version or (B) for the white-box version")
        self.di = di
        self.ti = ti
        if variant == "A":
            self.has_surrogate = True

    def set_model(self, model, model_name):
        self.model_name = model_name
        model = model.to(device)
        self.attack = Admix(model, normalize=None, device=device, eps=epsilon[self.dataset_name], steps=10, alpha=1.6,
                            decay=1.0, portion=0.2, size=3,
                            num_classes=(1000 if self.dataset_name == "imagenet" else 10), targeted=False, di=self.di,
                            ti=self.ti)

    def forward(self, x, y):
        return self.attack(x, y)

    def base_name(self):
        return "Admix"

    def name(self):
        name = ""
        if self.di:
            name += "DI-"
        if self.ti:
            name += "TI-"
        name += "Admix"

        return name
