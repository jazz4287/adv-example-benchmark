from attacks.attack import Attack


class BenignAttack(Attack):
    def __init__(self, variant: str, threat_model, dataset_name, *args, **kwargs):
        super().__init__(variant, threat_model, dataset_name)

    def forward(self, x, y):
        return x

    def name(self):
        return "Benign"
