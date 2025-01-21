import torchvision.transforms

from attacks.attack import Attack
from attacks.ssah.original_ssah import SSAH
from global_settings import device


class SSAHAttack(Attack):
    variants = ["A", "B"]
    params = {"cifar10": {"wave": "haar",
                          "num_iteration": 150,
                          "learning_rate": 0.001,
                          "m": 0.2,
                          "alpha": 1.0,
                          "lambda_lf": 0.1,
                          "Targeted": False},
              "imagenet": {"wave": "haar",
                           "num_iteration": 200,
                           "learning_rate": 0.0001,
                           "m": 0.2,
                           "alpha": 1.0,
                           "lambda_lf": 0.1,
                           "Targeted": False}
    }

    def __init__(self, variant, threat_model, dataset_name, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        assert variant is not None, (f"Please specify a variant for the SSAH attack. Either (B) for the transferable"
                                     f" version or (A) for the white-box version")
        if self.variant == "B":
            self.has_surrogate = True
        self.attack = None
        self.model_name = None
        self.experiment = False

    def set_model(self, model, model_name):
        model = model.to(device)
        self.model_name = model_name
        self.attack = SSAH(model=model, model_name=model_name, dataset=self.dataset_name, device=device,
                           **self.params[self.dataset_name])

    def forward(self, x, y):
        original_input_size = None
        if self.dataset_name == "imagenet" and self.variant == "B" and not self.experiment:
            #     we need to reshape the input image to the size of
            original_input_size = x.size()[-2:]
            x = torchvision.transforms.Resize((32, 32))(x)

        x = self.attack(x)
        if original_input_size is not None:
            x = torchvision.transforms.Resize(original_input_size)(x)
        return x

    def name(self):
        return "SSAH"
