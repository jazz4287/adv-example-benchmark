from attacks.attack import Attack
from attacks.square.original_square import Square
from global_settings import epsilon, device

class SquareAttack(Attack):
    def __init__(self, variant, threat_model, dataset_name, seed=None, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)
        self.attack = None
        self.model_name = None
        assert seed is not None, "please specify the seed when initializing the attack"
        self.seed = seed

    def set_model(self, model, model_name):
        self.model_name = model_name
        model = model.to(device)
        # hyperparameters from the paper that yield the best performance while considering runtime.
        # num_queries = 20000  # number from their analysis against robust models
        num_queries = 5000  # we would want to use 20k, but that's just too much in terms of runtime. Would take months on multiple A100 gpus to get all model results.
        num_restarts = 50  # same as their paper, we add a early stop to compensate
        p = 0.05 if self.dataset_name == "imagenet" else 0.3  # as per their paper
        loss = "margin"  # in their code they use the margin loss when the attack is untargeted and the ce loss otherwise (attack.py line 282: args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'
        early_stop = 3  # we extended their code with an early stopping system in case the attack can't improve (successfully attack one more image compared to before) for {early_stop} restarts, we stop to save computation
        self.attack = Square(model=model, eps=epsilon[self.dataset_name], norm=self.threat_model, n_queries=num_queries,
                             n_restarts=num_restarts, p_init=p, seed=self.seed, resc_schedule=True, loss=loss,
                             verbose=False, early_stop=early_stop)
        self.attack.targeted = False


    def forward(self, x, y):
        return self.attack(x, y)

    def name(self):
        return "Square"

