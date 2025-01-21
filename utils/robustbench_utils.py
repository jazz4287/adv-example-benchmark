# duplicate file to load ChenWRN because they messed up for that model in their master's branch implementation
import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import requests
import timm
import torch
from torch import nn
import gdown

from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import DATASET_CLASSES, download_gdrive_new, rm_substr_from_state_dict, add_substr_to_state_dict, _safe_load_state_dict


def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
               BenchmarkDataset] = BenchmarkDataset.cifar_10,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               custom_checkpoint: str = "",
               norm: Optional[str] = None) -> nn.Module:
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it
      overrides ``threat_model``

    :return: A ready-to-used trained model.
    """
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    if norm is None:
        # since there is only `corruptions` folder for models in the Model Zoo
        threat_model = ThreatModel(threat_model).value.replace('_3d', '')
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    lower_model_name = model_name.lower().replace('-', '_')
    timm_model_name = f"{lower_model_name}_{dataset_.value.lower()}_{threat_model_.value.lower()}"

    if timm.is_model(timm_model_name):
        return timm.create_model(timm_model_name,
                                 num_classes=DATASET_CLASSES[dataset_],
                                 pretrained=True,
                                 checkpoint_path=custom_checkpoint).eval()

    model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
    model_path = model_dir_ / f'{model_name}.pt'

    models = all_models[dataset_][threat_model_]

    if models[model_name]['gdrive_id'] is None:
        raise ValueError(
            f"Model `{model_name}` nor {timm_model_name} aren't a timm model and has no `gdrive_id` specified."
        )

    if not isinstance(models[model_name]['gdrive_id'], list):
        model = models[model_name]['model']()
        if dataset_ == BenchmarkDataset.imagenet and 'Standard' in model_name:
            return model.eval()

        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive_new(models[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if 'Kireev2021Effectiveness' in model_name or model_name == 'Andriushchenko2020Understanding':
            checkpoint = checkpoint[
                'last']  # we take the last model (choices: 'last', 'best')
        try:
            # needed for the model of `Carmon2019Unlabeled`
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
            # needed for the model of `Chen2020Efficient`
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')

        if dataset_ == BenchmarkDataset.imagenet:
            # Adapt checkpoint to the model defition in newer versions of timm.
            if model_name in [
                'Liu2023Comprehensive_Swin-B',
                'Liu2023Comprehensive_Swin-L',
            ]:
                try:
                    from timm.models.swin_transformer import checkpoint_filter_fn
                    state_dict = checkpoint_filter_fn(state_dict, model.model)
                except:
                    pass

            # Some models need input normalization, which is added as extra layer.
            if model_name not in [
                'Singh2023Revisiting_ConvNeXt-T-ConvStem',
                'Singh2023Revisiting_ViT-B-ConvStem',
                'Singh2023Revisiting_ConvNeXt-S-ConvStem',
                'Singh2023Revisiting_ConvNeXt-B-ConvStem',
                'Singh2023Revisiting_ConvNeXt-L-ConvStem',
                'Peng2023Robust',
                'Chen2024Data_WRN_50_2',   # the only thing we do is fix a typo here
            ]:
                state_dict = add_substr_to_state_dict(state_dict, 'model.')

        model = _safe_load_state_dict(model, model_name, state_dict, dataset_)

        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial, Diffenderfer2021Winning_LRR_CARD_Deck)
    else:
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        for i, gid in enumerate(models[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive_new(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i),
                                    map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(
                    checkpoint['state_dict'], 'module.')
            except KeyError:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

            if model_name.startswith('Bai2023Improving'):
                # TODO: make it cleaner.
                if i < 2:
                    model.comp_model.models[i] = _safe_load_state_dict(
                        model.comp_model.models[i], model_name, state_dict, dataset_)
                    model.comp_model.models[i].eval()
                else:
                    model.comp_model.policy_net = _safe_load_state_dict(
                        model.comp_model.policy_net, model_name, state_dict['model'], dataset_)
                    model.comp_model.bn = _safe_load_state_dict(
                        model.comp_model.bn, model_name, state_dict['bn'], dataset_)
            elif model_name.startswith('Bai2024MixedNUTS'):
                if i == 0:
                    model.std_model = _safe_load_state_dict(
                        model.std_model, model_name, state_dict, dataset_)
                elif i == 1:
                    if dataset_ == BenchmarkDataset.imagenet:
                        from timm.models.swin_transformer import checkpoint_filter_fn
                        state_dict = checkpoint_filter_fn(state_dict, model.rob_model.model)
                        state_dict = add_substr_to_state_dict(state_dict, 'model.')
                    model.rob_model = _safe_load_state_dict(
                        model.rob_model, model_name, state_dict, dataset_)
                else:
                    raise ValueError('Unexpected checkpoint.')
            else:
                model.models[i] = _safe_load_state_dict(model.models[i],
                                                        model_name, state_dict,
                                                        dataset_)
                model.models[i].eval()

        return model.eval()
