from attacks.attack import Attack
from art.attacks.evasion import AutoConjugateGradient
from global_settings import device, epsilon
from art.estimators.classification import PyTorchClassifier
import torch
from attacks.acg.utils.Gather import GatherManager
from attacks.acg.original_acg import AUTOConjugate
from attacks.acg.utils import read_yaml
from addict import Dict as AttrDict


class ACGAttack(Attack):
    def __init__(self, variant, threat_model, dataset_name, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        self.attack = None
        self.model_name = None

    def set_model(self, model, model_name):
        model = model.to(device)
        self.model_name = model_name
        # input_shape = (3, 224, 224) if self.dataset_name == "imagenet" else (3, 32, 32)
        # n_classes = 1000 if self.dataset_name == "imagenet" else 10
        # estimator = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(),
        #                               optimizer=torch.optim.SGD(model.parameters(), lr=0.01), input_shape=input_shape,
        #                               nb_classes=n_classes)
        # self.attack = AutoConjugateGradient(estimator=estimator, norm="inf", eps=epsilon[self.dataset_name],
        #                                     targeted=False, verbose=False)
        gather = GatherManager().getGather(model_name)
        # self._reset_gather_manager()
        yaml_paths = ["attacks/acg/di.yaml"]
        if self.dataset_name == "cifar10":
            yaml_paths.append("attacks/acg/cifar10.yaml")
        else:
            yaml_paths.append("attacks/acg/imagenet.yaml")
        config = AttrDict(read_yaml(yaml_paths))
        config.norm = self.threat_model
        self.attack = AUTOConjugate(model, W=None, bias=None, bound_opt=None,
                                    config=config, gather=gather, save_image=False, experiment=False,
                                    export_statistics=False, epsilon=epsilon[self.dataset_name])

    # def _reset_gather_manager(self):
    #     GatherManager.listofGather = dict()
    #     gather_manager = GatherManager()
    #     gather = gather_manager.getGather(self.model_name)
    #     gather.setParam("epsilon", epsilon[self.dataset_name])
    #     if self.attack is not None:
    #         self.attack.gather = gather


    def forward(self, x, y):
        # self._reset_gather_manager()  # otherwise it throws an error if you use it more than once
        # return torch.Tensor(self.attack.generate(x=x.cpu().numpy(), y=y.cpu().numpy())).to(device)
        adv_x, _ = self.attack.attack(x, y)
        return adv_x

    def name(self):
        return "ACG"
