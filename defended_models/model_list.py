# we only care about cifar10 l_inf and imagenet l_inf for now
import os
from robustbench.model_zoo.models import model_dicts
from robustbench.model_zoo.enums import ThreatModel, BenchmarkDataset


def list_model_names(model_folder:  str, dataset: str = "cifar10", threat_model: str = "Linf"):
    assert dataset in ["cifar10", "cifar100", "imagenet"]
    assert threat_model in ["Linf", "L2", "L0"]
    threat_model = ThreatModel(threat_model)
    dataset = BenchmarkDataset(dataset)
    models = list(model_dicts[dataset][threat_model].keys())
    return models
