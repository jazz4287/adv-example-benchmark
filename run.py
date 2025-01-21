
import pyximport
pyximport.install()

from attacks.lgv import LGVAttack
from attacks.benign import BenignAttack
from attacks.ssah import SSAHAttack
from attacks.acg import ACGAttack
from attacks.aa import AutoAttack
from attacks.aaa import AdaptiveAutoAttack
from attacks.admix import AdmixAttack
from attacks.vnifgsm import VNIFGSMAttack
from attacks.bia import BIAAttack
from attacks.square import SquareAttack
from global_settings import (device, epsilon, not_working_models, undefended_model_names, ssah_save_path,
                             has_not_working_models, worst_and_best_model_names, model_paths)
from utils.utils import set_seeds
from utils.models import load_pytorch_model, EnsembleModel, load_defended_or_undefended
import torch.backends.cudnn as cudnn
from defended_models.model_list import list_model_names
import os
import torch
from collections import OrderedDict


attack_list = {
                # "pixle": PixleAttack,
                "lgv": LGVAttack,
                "benign": BenignAttack,
                "ssah": SSAHAttack,
                "autoattack": AutoAttack,
                # "aaa": AdaptiveAutoAttack,
                "acg": ACGAttack,
                "admix": AdmixAttack,
                "vnifgsm": VNIFGSMAttack,
                "bia": BIAAttack,
                "square": SquareAttack
}




def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument('-a', '--attack', default=None, type=str, choices=list(attack_list.keys()))
    parser.add_argument("-d", "--dataset", default=None, type=str, choices=["cifar10", "imagenet"])
    parser.add_argument("-t", "--threat-model", default=None, type=str, choices=["Linf", "L0"])
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-v", "--variant", default=None, type=str)
    parser.add_argument("-s", "--seed", default=1337, type=int)
    parser.add_argument("-m", "--mode", default=None, choices=["defended", "undefended"], type=str)
    parser.add_argument("-e", "--experiment", default=None, choices=["worst_defended", "best_defended"])
    parser.add_argument("-c", "--combination", default=None, type=str, choices=["di", "di-ti", "da", "rn"])
    parser.add_argument("-r", "--reverse", default=False, action="store_true")
    return parser


def load_attack(args):
    if args.experiment is None:
        if args.attack == "lgv":
            attack = LGVAttack(args.variant, args.threat_model, args.dataset, di=(args.combination == "di" or args.combination == "di-ti"),
                                   ti=(args.combination == "di-ti"), seed=args.seed)
            if args.variant == "A":
                # use a pre-trained resnet50 as mentioned in the paper.
                assert attack.has_surrogate, f"Supposed to be the surrogate version, verify that the variant is properly set in both the args and the class"
                surrogate_model_name = "resnet50"
                surrogate_model = load_pytorch_model(surrogate_model_name, args.dataset)

                attack.set_model(surrogate_model, surrogate_model_name)
        elif args.attack == "ssah":
            attack = SSAHAttack(args.variant, args.threat_model, args.dataset, seed=args.seed)
            if args.variant == "B":
                # We use the resnet20 cifar100 model that they use in their paper since for them,
                # it yielded the best results in their experiments compared to the cifar10 one and also we can use it
                # for both imagenet and cifar10 since there is supposedly no overlap?
                assert attack.has_surrogate, f"Supposed to be the surrogate version, verify that the variant is properly set in both the args and the class"
                from attacks.ssah.resnet import ResNet as ssah_resnet
                # where you should put their model checkpoint that you can find on the google drive mentioned in their repo
                path = os.path.join(ssah_save_path, "cifar100-r20.pth.tar")
                checkpoint = torch.load(path)
                surrogate_model_name = "resnet20"
                surrogate_model = ssah_resnet(20, 100)
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if 'module.' in k:
                        name = k[7:]  # remove `module.`
                    else:
                        name = k
                    new_state_dict[name] = v
                surrogate_model.load_state_dict(new_state_dict)
                surrogate_model.eval()

                attack.set_model(surrogate_model, surrogate_model_name)
        elif args.attack == "admix":
            attack = AdmixAttack(args.variant, args.threat_model, args.dataset, di=(args.combination == "di" or args.combination == "di-ti"),
                                   ti=(args.combination == "di-ti"), seed=args.seed)
            if args.variant == "A":
                # they report their best result with an ensemble-model
                # they use four models for imagenet: Inception-v3, Inception-v4, IncRes-v2 and ResNet101
                # we do not have these models for cifar10, so as such, we supplement it with a diverse set of undefended models
                if args.dataset == "cifar10":
                    model_names = ["googlenet", "resnet50", "vgg", "mobilenetv2"]
                else:
                    # no inception v4 so we replace it with googlenet. no incres-v2 so we replace it with vgg
                    model_names = ["inception_v3", "googlenet", "resnet101", "vgg19"]
                models = [load_pytorch_model(model_name, args.dataset) for model_name in model_names]
                surrogate_model = EnsembleModel(models).to(device)
                attack.set_model(surrogate_model, "+".join(model_names))
        elif args.attack == "vnifgsm":
            attack = VNIFGSMAttack(args.variant, args.threat_model, args.dataset, di=(args.combination == "di" or args.combination == "di-ti"),
                                   ti=(args.combination == "di-ti"), seed=args.seed)
            # same as admix
            if args.variant == "A":
                # they report their best result with an ensemble-model
                # they use four models for imagenet: Inception-v3, Inception-v4, IncRes-v2 and ResNet101
                # we do not have these models for cifar10, so as such, we supplement it with a diverse set of undefended models
                if args.dataset == "cifar10":
                    model_names = ["googlenet", "resnet50", "vgg", "mobilenetv2"]
                else:
                    # no inception v4 so we replace it with googlenet. no incres-v2 so we replace it with vgg
                    model_names = ["inception_v3", "googlenet", "resnet101", "vgg19"]
                models = [load_pytorch_model(model_name, args.dataset) for model_name in model_names]
                surrogate_model = EnsembleModel(models).to(device)
                attack.set_model(surrogate_model, "+".join(model_names))
        elif args.attack == "bia":
            # for their augmentations, using both "da" and "rn" does not work well together even once in their experiments
            # so we don't consider it
            attack = BIAAttack(args.variant, args.threat_model, args.dataset,
                                              da=(args.combination == "da"),
                                              rn=(args.combination == "rn"), seed=args.seed)
            if args.variant == "A":
                # for surrogate model, they have an option of 4 possible models to use and unlike other transferable attacks
                # in a similar situation, they do not have a clear winner. So we instead train a generator against the
                # ensemble of the 4 models they provide.
                model_names = ["vgg16", "vgg19", "resnet152", "densenet169"]
                # they use fixed-dataset transferability: only imagenet
                models = [load_pytorch_model(model_name, dataset_name="imagenet") for model_name in model_names]
                surrogate_model = EnsembleModel(models).to(device)
                attack.set_model(surrogate_model, "+".join(model_names))
        elif args.attack == "autoattack":
            attack = AutoAttack(args.variant, args.threat_model, args.dataset, di=(args.combination == "di" or args.combination == "di-ti"),
                                   ti=(args.combination == "di-ti"), da=(args.combination == "da"), rn=(args.combination == "rn"),
                                seed=args.seed)
            if args.variant == "T":
                if args.dataset == "cifar10":
                    model_names = ["googlenet", "resnet50", "vgg", "mobilenetv2"]
                else:
                    # no inception v4 so we replace it with googlenet. no incres-v2 so we replace it with vgg
                    model_names = ["inception_v3", "googlenet", "resnet101", "vgg19"]
                models = [load_pytorch_model(model_name, args.dataset) for model_name in model_names]
                surrogate_model = EnsembleModel(models).to(device)
                attack.set_model(surrogate_model, "+".join(model_names))
        else:
            attack = attack_list[args.attack](args.variant, args.threat_model, args.dataset, di=(args.combination == "di" or args.combination == "di-ti"),
                                   ti=(args.combination == "di-ti"), da=(args.combination == "da"),
                                              rn=(args.combination == "rn"), seed=args.seed)
    else:
        attack = attack_list[args.attack](args.variant, args.threat_model, args.dataset, di=(args.combination == "di" or args.combination == "di-ti"),
                                   ti=(args.combination == "di-ti"), da=(args.combination == "da"),
                                              rn=(args.combination == "rn"), seed=args.seed)
        assert attack.has_surrogate, f"The attack is supposed to be transferable attack to use the surrogate models for the experiment"
        model_name = worst_and_best_model_names[args.dataset][args.experiment]
        print(f"Experiment: {args.experiment}, using surrogate model: {model_name}")
        surrogate_model = load_defended_or_undefended(dataset_name=args.dataset, model_name=model_name,
                                                      threat_model=args.threat_model, model_dir=model_paths)
        attack.set_model(surrogate_model, model_name)
        if args.attack == "ssah":
            attack.experiment = True  # set flag to True to bypass the reshape for imagenet

    return attack


def main(args):

    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.seed is not None:
        set_seeds(args.seed)
    assert args.attack is not None
    assert args.dataset is not None
    assert args.threat_model is not None

    attack = load_attack(args).to(device)
    if args.variant is None:
        args.variant = ""
    print(args)
    print(f"Epsilon: {epsilon[args.dataset]}")
    if args.mode == "defended":
        model_names = list_model_names(model_folder="./defended_models/model_info/", dataset=args.dataset,
                                       threat_model=args.threat_model)
    else:
        model_names = undefended_model_names[args.dataset]
    if args.reverse:
        enumerator = enumerate(reversed(model_names))
    else:
        enumerator = enumerate(model_names)
    for model_num, model_name in enumerator:
        if args.reverse:
            print(f"Attacking model #{len(model_names) - model_num}/{len(model_names)} (reversed): {model_name}")
        else:
            print(f"Attacking model #{model_num+1}/{len(model_names)}: {model_name}")
        if "".join([args.dataset, args.attack, args.variant]) in has_not_working_models:
            if (model_name in not_working_models[args.dataset][args.attack][args.variant]):
                print(f"Does not load or run properly, skipping.")
                continue
        attack.run_attack(model_name=model_name,
                          dataset_name=args.dataset,
                          threat_model=args.threat_model,
                          use_robust_5000=True,
                          overwrite_results=False,
                          batch_size=args.batch_size,
                          experiment=args.experiment)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
