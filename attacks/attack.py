import torch
import torch.nn as nn
from global_settings import device, log_dir, epsilon, model_paths, worst_and_best_model_names
from tqdm import tqdm
from utils.models import load_defended_or_undefended
from utils.data import load_data, get_preprocessing
import os
import json


class Attack(nn.Module):
    variants = []
    combinations = {}
    def __init__(self, variant: str, threat_model: str, dataset_name: str, *args, **kwargs):
        super().__init__()
        self.variant = variant
        self.threat_model = threat_model
        self.has_surrogate = False
        self.dataset_name = dataset_name
        self.sur_cache = None


    def set_model(self, model, model_name):
        pass

    def load_results(self, dataset_name, threat_model, model_name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # TODO: fix logging so that multiple processes can log at the same time without overwriting each other
        # log the results
        if os.path.exists(os.path.join(log_dir, "results.json")):
            with open(os.path.join(log_dir, "results.json"), "r") as f:
                results = json.load(f)
            if results.get(dataset_name, None) is None:
                results[dataset_name] = {threat_model: {str(self): {model_name: {}}}}
            elif results[dataset_name].get(threat_model, None) is None:
                results[dataset_name][threat_model] = {str(self): {model_name: {}}}
            elif results[dataset_name][threat_model].get(str(self), None) is None:
                results[dataset_name][threat_model][str(self)] = {model_name: {}}
            elif results[dataset_name][threat_model][str(self)].get(model_name, None) is None:
                results[dataset_name][threat_model][str(self)][model_name] = {}
        else:
            results = {dataset_name: {threat_model: {str(self): {model_name: {}}}}}

        return results

    def keep_in_bounds(self, adv_x, x):
        pert = adv_x - x
        if self.threat_model == "Linf":
            pert = torch.clamp(pert, min=-epsilon[self.dataset_name], max=epsilon[self.dataset_name])
        else:
            raise NotImplementedError
        new_adv_x = torch.clamp(x+pert, min=0, max=1)
        return new_adv_x

    def attack_model(self, model, loader, post_processing=None, *args, **kwargs):
        model = model.to(device)
        successes = 0
        failures = 0
        bar = tqdm(loader)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            adv_x = self.forward(x, y, *args, **kwargs)
            # keep within the epsilon bound
            adv_x = self.keep_in_bounds(adv_x, x)
            if post_processing is not None:
                adv_x = post_processing(adv_x)
            with torch.no_grad():
                pred = model(adv_x)
            adv_y = pred.argmax(dim=1)
            successes += (y == adv_y).to(torch.int).sum()
            failures += (y != adv_y).to(torch.int).sum()
            bar.set_description(f"{str(self)} | Successes: {successes} | Failures: {failures} | Adv Acc.: {successes / (successes+failures):.2f} | ASR: {failures / (successes+failures):.2f}")

        return successes, failures

    def run_attack(self, model_name: str, dataset_name: str, threat_model: str = "Linf", use_robust_5000: bool = True,
                   overwrite_results: bool = False, batch_size: int = 64, experiment: str = None):
        results = self.load_results(dataset_name, threat_model, model_name)
        exp = ""
        if experiment is not None:
            surrogate_model_name = worst_and_best_model_names[dataset_name][experiment]
            exp += experiment + "_" + surrogate_model_name + "_"
        exp += "robust_5000_adv_acc" if dataset_name == "imagenet" and use_robust_5000 else "full_adv_acc"
        if results[dataset_name][threat_model][str(self)][model_name].get(exp, None) is not None and not overwrite_results:
            # no need to run it
            return None, None
        if model_name == "models_foun":
            return None, None
        try:
            model = load_defended_or_undefended(model_name=model_name, dataset_name=dataset_name, threat_model=threat_model,
                               model_dir=model_paths).to(device)
        except (KeyError, RuntimeError) as e:
            # we now try to see if it's the model was not properly saved
            # we try to load the downloaded checkpoint
            print(e)
            print(f"No model named {model_name}")
            return None, None
        if not self.has_surrogate:
            self.set_model(model, model_name)
            loader = load_data(train=False, threat_model=threat_model, model_name=model_name,
                               dataset_name=dataset_name, use_robust_5000=use_robust_5000, batch_size=batch_size)
            post_processing = None
        else:
            loader = load_data(train=False, threat_model=threat_model, model_name=self.model_name,
                               dataset_name=dataset_name, use_robust_5000=use_robust_5000, batch_size=batch_size)
            post_processing = get_preprocessing(dataset_name=dataset_name, threat_model=threat_model,
                                                model_name=model_name)
        if self.has_surrogate:
            # we only need to generate the adversarial examples once since they are independent of the model being attacked
            if self.sur_cache is None:
                print(f"Generating surrogate output cache")
                self.sur_cache = self.generate_adv_examples(loader)
            print(f"Using surrogate output cache")
            cached_loader = torch.utils.data.DataLoader(self.sur_cache, batch_size=batch_size, shuffle=False)
            successes, failures = self.get_accuracy(model, cached_loader)
        else:
            successes, failures = self.attack_model(model, loader, post_processing)

        # we reload the results in case some other process has updated them
        results = self.load_results(dataset_name, threat_model, model_name)
        results[dataset_name][threat_model][str(self)][model_name][exp] = float(successes/(successes + failures))
        with open(os.path.join(log_dir, "results.json"), "w+") as f:
            json.dump(results, f, indent=4)
        return successes, failures

    def generate_adv_examples(self, loader):
        adv_xs = []
        adv_ys = []
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            adv_x = self.forward(x, y)
            # keep within the epsilon bound
            adv_x = self.keep_in_bounds(adv_x, x)
            adv_xs.append(adv_x.cpu())
            adv_ys.append(y.cpu())
        adv_xs = torch.cat(adv_xs, dim=0)
        adv_ys = torch.cat(adv_ys, dim=0)
        return torch.utils.data.TensorDataset(adv_xs, adv_ys)

    def get_accuracy(self, model, loader, post_processing=None):
        model = model.to(device)
        model.eval()
        successes = 0
        failures = 0
        for adv_x, y in loader:
            adv_x, y = adv_x.to(device), y.to(device)
            if post_processing is not None:
                adv_x = post_processing(adv_x)
            with torch.no_grad():
                pred = model(adv_x)
            adv_y = pred.argmax(dim=1)
            successes += (y == adv_y).to(torch.int).sum()
            failures += (y != adv_y).to(torch.int).sum()
        return successes, failures

    def name(self):
        return "Attack"

    def base_name(self):
        # for attacks with combinations
        return self.name()

    def __str__(self):
        if self.variant is not None:
            return f"{self.name()} ({self.variant})"
        else:
            return f"{self.name()}"
