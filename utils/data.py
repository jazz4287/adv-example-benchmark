from robustbench.data import PREPROCESSINGS, load_imagenet
import torchvision.datasets as datasets
from global_settings import data_paths
import torch.utils.data as data
import os
from robustbench.model_zoo.models import model_dicts
from robustbench.model_zoo.enums import ThreatModel, BenchmarkDataset
import torchvision


def get_preprocessing(dataset_name, threat_model, model_name):
    threat_model = ThreatModel(threat_model)
    dataset_name = BenchmarkDataset(dataset_name)
    if dataset_name == BenchmarkDataset.cifar_10:
        preprocessing = PREPROCESSINGS[None]
    elif dataset_name == BenchmarkDataset.cifar_100:
        preprocessing = PREPROCESSINGS[None]
    elif dataset_name == BenchmarkDataset.imagenet:
        preprocessing = model_dicts[dataset_name][threat_model].get(model_name, None)
        if preprocessing is not None:
            preprocessing = model_dicts[dataset_name][threat_model][model_name].get('preprocessing', None)
        preprocessing = PREPROCESSINGS['Res256Crop224'] if preprocessing is None else PREPROCESSINGS[preprocessing]
    else:
        raise NotImplementedError
    return preprocessing


def load_data(dataset_name: str, threat_model: str, model_name, train: bool = False, use_robust_5000: bool = True,
              batch_size: int = 64, preprocessing=None):
    """
    Load dataset
    :param dataset_name: name of the dataset
    :param train: whether to use the training or testing dataset
    :param use_robust_5000: Only applies to imagenet. If True, loads directly the same images as robustbench
    :return: return a dataloader that yields x,y pairs.
    """
    if preprocessing is None:
        preprocessing = get_preprocessing(dataset_name, threat_model, model_name)
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=data_paths,
                                   train=train,
                                   transform=preprocessing,
                                   download=True)
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=data_paths,
                                    train=train,
                                    transform=preprocessing,
                                    download=True)

    elif dataset_name == "imagenet":
        if use_robust_5000 and not train:
            x, y = load_imagenet(n_examples=5000, data_dir=os.path.join(data_paths, "imagenet"),
                                 transforms_test=preprocessing)
            dataset = data.TensorDataset(x, y)
        else:
            dataset = datasets.ImageFolder(
                os.path.join(data_paths, "imagenet", "train" if train else "val"),
                preprocessing
            )

    else:
        raise NotImplementedError
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=train,
                             num_workers=16)

    return loader
