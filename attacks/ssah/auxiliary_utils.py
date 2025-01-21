from torchvision import datasets, transforms
import numpy as np
import random
import torch
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def normalize_fn(dataset):
    if dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])

    return normalize


def predict(model, inputs, opt):
    with torch.no_grad():
        outputs = model(normalize_fn(opt.dataset)(inputs))
        pred = outputs.max(1, keepdim=False)[1]
        return pred


def common(targets, pred):
    common_id = np.where(targets.cpu() == pred.cpu())[0]
    return common_id


def attack_success(targets, pred):
    attack_id = np.where(targets.cpu() != pred.cpu())[0]
    return attack_id


def load_cifar10(opt):
    path = os.path.join(opt.dataset_root, 'cifar10/')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=path,
                               train=False,
                               transform=transform,download=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=opt.bs,
                                             shuffle=True,
                                             num_workers=opt.workers,
                                             pin_memory=True)
    return dataloader, len(dataset)


def load_cifar100(opt):
    path = os.path.join(opt.dataset_root, 'cifar-100-python/')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=path,
                                train=False,
                                transform=transform,download=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=opt.bs,
                                             shuffle=True,
                                             num_workers=opt.workers,
                                             pin_memory=True)
    return dataloader, len(dataset)


def load_imagenet_val(opt):
    path = os.path.join(opt.dataset_root, 'ILSVRC2012/val/')
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=path,
                                   transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=opt.bs,
                                             shuffle=True,
                                             num_workers=opt.workers,
                                             pin_memory=True
                                             )
    return dataloader, len(dataset)


def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message
