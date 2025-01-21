import torch
from pytorch_cifar.main import model_dict
import torchvision
from torchvision.models import list_models


epsilon = {"cifar10": float(8/255), "imagenet": float(4/255)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model_paths = "./models/"
data_paths = "./datasets/"
log_dir = "./results/"
cifar10_checkpoint_dir = "./models/checkpoint/"
not_working_models = {"cifar10":
                          {"lgv":
                               {"B": ["Kang2021Stable", "Bai2024MixedNUTS"]},
                           "all": {"any": ["preactresnet18"]}},
                      "imagenet":
                          {"lgv":
                               {"B": ["Bai2024MixedNUTS"]},
                           "admix": {"B": ["efficientnet_b6", "efficientnet_b7", "efficientnet_v2_l", "regnet_y_128gf",
                                           "vit_h_14"]}}}  # can't run them on 80GB gpus with batch size >=3, admix requires 3+
surrogate_models_used = {  # use upper case for easier use with the notebooks.
    "LGV (A)": {"cifar10": ["resnet50"],
                  "imagenet": ["resnet50"]},
    "ADMIX (A)": {"cifar10": ["googlenet", "resnet50", "vgg", "mobilenetv2"],
                    "imagenet": ["inception_v3", "googlenet", "resnet101", "vgg19"]},
    "VNIFGSM (A)":  {"cifar10": ["googlenet", "resnet50", "vgg", "mobilenetv2"],
                    "imagenet": ["inception_v3", "googlenet", "resnet101", "vgg19"]},
    "BIA (A)": {"imagenet": ["vgg16", "vgg19", "resnet152", "densenet169"]}

}  # ssah does not use a model that we evaluate against.
has_not_working_models = ["cifar10"+"lgv"+"B", "imagenet"+"lgv"+"B", "imagenet"+"admix"+"B"]
undefended_model_names = {"cifar10": list(model_dict.keys()), "imagenet": list_models(module=torchvision.models)}
lgv_save_path = "./lgv"
ssah_save_path = "./ssah"
worst_and_best_model_names = {"cifar10": {"best_defended": "Peng2023Robust", # fourth but the first one on the leaderboard that is in the ModelZoo
                                          "worst_defended": "Ding2020MMA"},
                              "imagenet": {"best_defended": "Liu2023Comprehensive_Swin-L",  # second but the first one on the leaderboard that is in the ModelZoo
                                           "worst_defended": "Salman2020Do_R18"}}  # need to watch out some other models use the same defense, but different architecture
bia_save_path = "./bia"
