'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import random
import numpy as np




def main():
    model_paths = "./"

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument("--model", choices=["vgg", "resnet18", "preactresnet18", "googlenet", "densenet121",
                                            "resnext29", "mobilenet", "mobilenetv2", "dpn92", "shufflenetg2", "senet18",
                                            "shufflenetv2", "efficientnet-b0", "regnetx_200mf", "simpledla", "resnet50"])
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    seed_num = args.seed
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    print(f"seed set to {seed_num}")



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    model_name = args.model

    # Model
    print('==> Building model..')
    net = model_dict[model_name]()

    name = model_name
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    model_path = os.path.join(model_paths, f'checkpoint/{name}_ckpt.pt')

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch, best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(os.path.join(model_paths, "checkpoint")):
                os.mkdir(os.path.join(model_paths, "checkpoint"))
            torch.save(state, model_path)
            best_acc = acc
        return best_acc

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        best_acc = test(epoch, best_acc)
        scheduler.step()


if __name__ == '__main__':
    from models import *
    from utils import progress_bar
    main()
else:
    from pytorch_cifar.models import *
    # from pytorch_cifar.utils import progress_bar


model_dict = {
    "vgg": VGG,
    "resnet18": ResNet18,
    # "preactresnet18": PreActResNet18,
    "googlenet": GoogLeNet,
    "densenet121": DenseNet121,
    "resnext29": ResNeXt29_2x64d,
    "mobilenet": MobileNet,
    "mobilenetv2": MobileNetV2,
    "dpn92": DPN92,
    "shufflenetg2": ShuffleNetG2,
    "senet18": SENet18,
    "shufflenetv2": ShuffleNetV2,
    "efficientnet-b0": EfficientNetB0,
    "regnetx_200mf": RegNetX_200MF,
    "simpledla": SimpleDLA,
    "resnet50": ResNet50
}


models_args = {
    "vgg": {"vgg_name": "VGG19"},
    "shufflenetv2": {"net_size": 1}
}