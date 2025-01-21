from attacks.attack import Attack
from global_settings import device, epsilon, bia_save_path
import torch
from attacks.bia.train import train
import os
from attacks.bia.generator import GeneratorResnet


class TrainArgs:
    def __init__(self, batch_size, epochs, lr, da, rn):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.DA = da
        self.RN = rn


class BIAAttack(Attack):
    variants = ["A", "B", "T-a", "T-b", "T-c", "T-d"]
    combinations = {"A": ["da", "rn"],
                    "T-a": ["da", "rn"],
                    "T-b": ["da", "rn"],
                    "T-c": ["da", "rn"],
                    "T-d": ["da", "rn"]}
    def __init__(self, variant, threat_model, dataset_name, da: bool = False, rn: bool = False, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        self.attack = None
        self.generator = None
        self.model_name = None
        self.test_variants = ["T-a", "T-b", "T-c", "T-d"]
        self.test_models = {"T-a": "res152", "T-b": "vgg16", "T-c": "vgg19", "T-d": "dense169"}
        assert variant is not None, (f"Please specify a variant for the BIA attack. Either (A) for the "
                                     f" transferable version or (B) for the for the white-box version")
        self.da = da
        self.rn = rn
        if variant == "A" or variant in self.test_variants:
            self.has_surrogate = True

        if self.variant in self.test_variants:
            self.load_test_generator()

    def load_test_generator(self):
        if self.rn and self.da:
            save_checkpoint_suffix = 'BIA+RN+DA'
        elif self.rn:
            save_checkpoint_suffix = 'BIA+RN'
        elif self.da:
            save_checkpoint_suffix = 'BIA+DA'
        else:
            save_checkpoint_suffix = 'BIA'
        generator = GeneratorResnet().to(device)
        model_path = os.path.join(bia_save_path, "saved_models", self.test_models[self.variant], f"netG_{save_checkpoint_suffix}_0.pth")
        generator.load_state_dict(torch.load(model_path, map_location=device))
        self.generator = generator
        self.generator.eval()

    def set_model(self, model, model_name):
        if self.variant in self.test_variants:
            return
        self.model_name = model_name
        model = model.to(device)
        # they only use 1 epoch to train their generators in their paper for imagenet
        # that's about ~ 80k batches of 16 images ~= 1.28M images
        # for cifar10, 1 epoch = 50k images. however, cifar10 is easier to learn so we'll go for 1/3rd of the amount of images
        # hence epochs ~=8
        if self.variant == "A":
            epochs = 1
        elif self.variant == "B":
            if self.dataset_name == "cifar10":
                epochs = 8
            elif self.dataset_name == "imagenet":
                epochs = 1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if self.rn and self.da:
            save_checkpoint_suffix = 'BIA+RN+DA'
        elif self.rn:
            save_checkpoint_suffix = 'BIA+RN'
        elif self.da:
            save_checkpoint_suffix = 'BIA+DA'
        else:
            save_checkpoint_suffix = 'BIA'
        dataset = "imagenet" if self.variant == "A" else "cifar10"
        save_checkpoint_dir = os.path.join(bia_save_path, 'saved_models/{}/{}'.format(dataset, model_name))
        save_path = os.path.join(save_checkpoint_dir, 'netG_{}_{}.pth'.format(save_checkpoint_suffix, epochs-1))

        # as per the paper
        batch_size = 16
        lr = 0.0002

        if self.dataset_name == "cifar10" and model_name == "Peng2023Robust":
            # can't use a batch size of 16 because OOM.
            # shouldn't matter too much, should still train on the same overall amount of samples
            batch_size = 2
        elif self.dataset_name == "imagenet" and model_name == "efficientnet_b7":
            batch_size = 8

        args = TrainArgs(batch_size=batch_size, epochs=epochs, lr=lr, da=self.da, rn=self.rn)

        if os.path.exists(save_path):
            # already trained the generator
            netG = GeneratorResnet().to(device)
            netG.load_state_dict(torch.load(save_path))
        else:
            # we need to train it
            dataset_name = "imagenet" if self.variant == "A" else self.dataset_name
            # we use the epsilon of the dataset that the model will be inferred on to make it fair,
            # rather than use the epsilon of the dataset that it is being trained on.
            netG = train(args, model, model_name, dataset_name, self.threat_model, eps=epsilon[self.dataset_name])
            # netG = GeneratorResnet().to(device)
            # netG.load_state_dict(torch.load(save_path))
        self.generator = netG
        self.generator.eval()

    def forward(self, x, y):
        eps = epsilon[self.dataset_name]
        with torch.no_grad():
            adv = self.generator(x)

        adv = torch.min(torch.max(adv, x - eps), x + eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        return adv

    def base_name(self):
        return "BIA"

    def name(self):
        name = "BIA"
        if self.rn:
            name += "+RN"
        if self.da:
            name += "+DA"
        return name
