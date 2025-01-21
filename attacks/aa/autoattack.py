from attacks.attack import Attack
from autoattack import AutoAttack as orig_auto_attack
from global_settings import device, epsilon, undefended_model_names


class AutoAttack(Attack):
    def __init__(self, variant, threat_model, dataset_name, seed, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        self.attack = None
        self.model_name = None
        self.seed = seed

        if self.variant == "T":
            self.has_surrogate = True


    def set_model(self, model, model_name):
        model = model.to(device)
        self.model_name = model_name
        # rand = ("rand" if ((self.dataset_name == "imagenet") and (model_name in undefended_model_names[self.dataset_name])) else "standard")
        # if rand:
        #     print(f"AutoAttack rand flag set")
        # for some reason, imagenet undefended models make AA think they're randomized, need to toggle random version
        self.attack = orig_auto_attack(model, norm='Linf', eps=epsilon[self.dataset_name], verbose=False, seed=self.seed)

    def forward(self, x, y):
        return self.attack.run_standard_evaluation(x, y, bs=x.size(0))

    def name(self):
        return "AutoAttack"
