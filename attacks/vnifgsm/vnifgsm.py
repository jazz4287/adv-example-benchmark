from attacks.attack import Attack
from global_settings import device, epsilon
from attacks.vnifgsm.original_vnifgsm import VNIFGSM


class VNIFGSMAttack(Attack):
    variants = ["A", "B"]
    combinations = {"A": ["di", "ti", "di-ti"]}
    def __init__(self, variant, threat_model, dataset_name, di: bool = False, ti: bool = False, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        self.attack = None
        self.model_name = None

        assert variant is not None, (f"Please specify a variant for the {self.name()} attack. Either (A) for the transferable"
                                     f" version or (B) for the white-box version")
        self.di = di
        self.ti = ti
        if variant == "A":
            self.has_surrogate = True

    def set_model(self, model, model_name):
        self.model_name = model_name
        model = model.to(device)
        self.attack = VNIFGSM(model, eps=epsilon[self.dataset_name], steps=10, alpha=1.6,
                            decay=1.0, N=20, beta=1.5, di=self.di, ti=self.ti)  # as per the paper

    def forward(self, x, y):
        return self.attack(x, y)

    def base_name(self):
        return "VNI-FGSM"

    def name(self):
        name = "VNI"
        if self.di:
            name += "-DI"
        if self.ti:
            name += "-TI"
        name += "-FGSM"

        return name
