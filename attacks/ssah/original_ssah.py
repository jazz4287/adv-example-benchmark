import torch
import torch.nn as nn
from attacks.ssah.auxiliary_utils import normalize_fn
import torch.nn.functional as F
from attacks.ssah.DWT import *
from torch.autograd import Variable
import torch.optim as optim


class NoLastLayerWrapper(nn.Module):
    special_treatment = {"ResNest": -1,  # avoiding their custom global avg pool layer that squashes the dimensions.
                         "Kang2021Stable": -2,
                         "Bai2024MixedNUTS": -2
                         }

    def __init__(self, model, model_name, module_list):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.module_list = module_list

        # apply hook to second to last module
        self.output = None
        self.test_new = False
        handles = []
        def add_hook(m):
            def forward_hook(module, input, output):
                self.output = output

            handles.append(m.register_forward_hook(forward_hook))
        self.model_name: str
        if self.model_name.startswith('Debenedetti2022Light_XCiT'):
            last_idx = 0
            for i, module in enumerate(module_list):
                if isinstance(module, torch.nn.Conv2d):
                    last_idx = i
            self.module_list[last_idx].apply(add_hook)
            return
        # if isinstance(self.module_list[-1], torch.nn.Linear) or isinstance(self.module_list[-1], torch.nn.BatchNorm1d)\
        #         or isinstance(self.module_list[-1], torch.nn.Dropout) or isinstance(self.module_list[-1], torch.nn.Identity)\
        #         or isinstance(self.module_list[-1], torch.nn.ReLU) or isinstance(self.module_list[-1], torch.nn.GELU)\
        #         or isinstance(self.module_list[-1], torch.nn.LayerNorm) or isinstance(self.module_list[-1], torch.nn.Flatten):
        #     # uses a linear layer to get the logits
        #     i = -1
        #     while (isinstance(self.module_list[i], torch.nn.Linear) or isinstance(self.module_list[i], torch.nn.BatchNorm1d) or
        #            isinstance(self.module_list[i], torch.nn.Dropout) or isinstance(self.module_list[i],
        #                                                                             torch.nn.Identity) \
        #            or isinstance(self.module_list[i], torch.nn.ReLU) or isinstance(self.module_list[i], torch.nn.GELU) \
        #            or isinstance(self.module_list[i], torch.nn.LayerNorm) or isinstance(self.module_list[i], torch.nn.Flatten)
        #     ):
        #         i -= 1
        #     if self.model_name in list(self.special_treatment.keys()):
        #         i += self.special_treatment[self.model_name]
        #     self.module_list[i].apply(add_hook)  # apply the hook to the one before
        # else:
        #     i = -2
        #     if self.model_name in list(self.special_treatment.keys()):
        #         i += self.special_treatment[self.model_name]
        #     self.module_list[i].apply(add_hook)  # apply the hook to the one before

        i =- 1
        while (not isinstance(self.module_list[i], torch.nn.Conv2d)) and (not isinstance(self.module_list[i], torch.nn.BatchNorm2d)) and \
            (not isinstance(self.module_list[i], torch.nn.InstanceNorm2d)):
            i -= 1
        if self.test_new:
            halfway_point = int((len(self.module_list) + i) / 2)
            self.module_list[halfway_point].apply(add_hook)
        else:
            self.module_list[i].apply(add_hook)

    def forward(self, x):
        _ = self.model(x)
        return self.output


class SSAH(nn.Module):
    """"
    Parameters:
    -----------

    """

    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 num_iteration: int = 150,
                 learning_rate: float = 0.001,
                 device: torch.device = torch.device('cuda'),
                 Targeted: bool = False,
                 dataset: str = 'cifar10',
                 m: float = 0.2,
                 alpha: float = 1,
                 lambda_lf: float = 0.1,
                 wave: str = 'haar',) -> None:
        super(SSAH, self).__init__()
        self.model = model
        if model.__class__.__name__ == 'ResNest':
            print(model_name)
        self.model_name = model_name
        self.device = device
        self.lr = learning_rate
        self.target = Targeted
        self.num_iteration = num_iteration
        self.dataset = dataset
        self.m = m
        self.alpha = alpha
        self.lambda_lf = lambda_lf

        # is not a reliable way to get the model weights in order
        # self.encoder_fea = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.encoder_fea = None

        # self.encoder_fea = nn.DataParallel(self.encoder_fea)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.model = nn.DataParallel(self.model)

        self.normalize_fn = normalize_fn(self.dataset)

        self.DWT = DWT_2D_tiny(wavename= wave)
        self.IDWT = IDWT_2D_tiny(wavename= wave)

    def set_encoder(self, inputs):

        modules = []
        handles = []
        def add_hook(m):
            def forward_hook(module, input):
                modules.append(module)

            handles.append(m.register_forward_pre_hook(forward_hook))

        def apply_hooks_to_leaf(model):
            children = list(model.children())
            for child in children:
                if len(list(child.children())) > 0:
                    apply_hooks_to_leaf(child)
                else:
                    child.apply(add_hook)
        apply_hooks_to_leaf(self.model.module)
        dummy = torch.rand_like(inputs).to(self.device)
        self.model(dummy)
        # remove the hooks
        for handle in handles:
            handle.remove()

        # we now create a wrapper for the model that takes in the model and adds a hook to the second to layer module
        # called and outputs the output of the layer rather than the last layer
        self.encoder_fea = nn.DataParallel(NoLastLayerWrapper(self.model, self.model_name, modules))

    def fea_extract(self, inputs: torch.Tensor) -> torch.Tensor:
        fea = self.encoder_fea(inputs)
        b, c, h, w = fea.shape
        fea = self.avg_pool(fea).view(b, c)
        return fea

    def cal_sim(self, adv, inputs):
        adv = F.normalize(adv, dim=1)
        inputs = F.normalize(inputs, dim=1)

        r, c = inputs.shape
        sim_matrix = torch.matmul(adv, inputs.T)
        mask = torch.eye(r, dtype=torch.bool).to(self.device)
        pos_sim = sim_matrix[mask].view(r, -1)
        neg_sim = sim_matrix.view(r, -1)
        return pos_sim, neg_sim

    def select_setp1(self, pos_sim, neg_sim):
        neg_sim, indices = torch.sort(neg_sim, descending=True)
        pos_neg_sim = torch.cat([pos_sim, neg_sim[:, -1].view(pos_sim.shape[0], -1)], dim=1)
        return pos_neg_sim, indices

    def select_step2(self, pos_sim, neg_sim, indices):
        hard_sample = indices[:, -1]
        ones = torch.sparse.torch.eye(neg_sim.shape[1]).to(self.device)
        hard_one_hot = ones.index_select(0, hard_sample).bool()
        hard_sim = neg_sim[hard_one_hot].view(neg_sim.shape[0], -1)
        pos_neg_sim = torch.cat([pos_sim, hard_sim], dim=1)
        return pos_neg_sim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.encoder_fea is None:
            self.set_encoder(inputs)
        with torch.no_grad():
            inputs_fea = self.fea_extract(self.normalize_fn(inputs))

        # low frequency component
        inputs_ll = self.DWT(inputs)
        inputs_ll = self.IDWT(inputs_ll)

        # changes of variables
        eps = 3e-7
        modifier = torch.arctanh(inputs * (2 - eps * 2) - 1 + eps)
        modifier = Variable(modifier, requires_grad=True)
        modifier = modifier.to(self.device)
        optimizer = optim.Adam([modifier], lr=self.lr)

        lowFre_loss = nn.SmoothL1Loss(reduction='sum')

        for step in range(self.num_iteration):
            optimizer.zero_grad()
            self.encoder_fea.zero_grad()

            adv = 0.5 * (torch.tanh(modifier) + 1)
            adv_fea = self.fea_extract(self.normalize_fn(adv))

            adv_ll = self.DWT(adv)
            adv_ll = self.IDWT(adv_ll)

            pos_sim, neg_sim = self.cal_sim(adv_fea, inputs_fea)
            # select the most dissimilar one in the first iteration
            if step == 0:
                pos_neg_sim, indices = self.select_setp1(pos_sim, neg_sim)

            # record the most dissimilar ones by indices and calculate similarity
            else:
                pos_neg_sim = self.select_step2(pos_sim, neg_sim, indices)

            sim_pos = pos_neg_sim[:, 0]
            sim_neg = pos_neg_sim[:, -1]

            w_p = torch.clamp_min(sim_pos.detach() - self.m, min=0)
            w_n = torch.clamp_min(1 + self.m - sim_neg.detach(), min=0)

            adv_cost = torch.sum(torch.clamp(w_p * sim_pos - w_n * sim_neg, min=0))
            lowFre_cost = lowFre_loss(adv_ll, inputs_ll)
            total_cost = self.alpha * adv_cost + self.lambda_lf * lowFre_cost

            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

        adv = 0.5 * (torch.tanh(modifier.detach()) + 1)
        return adv
