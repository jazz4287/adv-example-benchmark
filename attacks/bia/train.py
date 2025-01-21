import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from attacks.bia.generator import GeneratorResnet
import os
from global_settings import device, bia_save_path
from utils.data import load_data
from tqdm import tqdm
from utils.models import EnsembleModel


class Extractor(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.ensemble = isinstance(model, EnsembleModel)  # if the model is an ensemble, we return the list of features
        self.model = model
        self.model_name = model_name
        self.output = []

    def set_encoder(self, inputs):
        if self.ensemble:
            models = self.model.models
        else:
            models = [self.model]

        for model in models:
            modules = []
            handles = []

            def add_hook(m):
                def forward_hook(module, input):
                    modules.append(module)

                handles.append(m.register_forward_pre_hook(forward_hook))

            def apply_hooks_to_leaf(in_model):
                children = list(in_model.children())
                for child in children:
                    if len(list(child.children())) > 0:
                        apply_hooks_to_leaf(child)
                    else:
                        child.apply(add_hook)

            apply_hooks_to_leaf(model)
            dummy = torch.rand_like(inputs).to(device)
            model(dummy)
            # remove the hooks
            for handle in handles:
                handle.remove()

            # finding the last layer part of the encoder
            handles = []
            def add_hook(m):
                def forward_hook(module, input, output):
                    self.output.append(output)

                handles.append(m.register_forward_hook(forward_hook))

            i = - 1
            while (not isinstance(modules[i], torch.nn.Conv2d)) and (
            not isinstance(modules[i], torch.nn.BatchNorm2d)) and \
                    (not isinstance(modules[i], torch.nn.InstanceNorm2d)):
                i -= 1
            # they look at intermediate features, so we don't take the end point of the feature pipeline but instead take
            # the halfway point
            # This unfortunately does not exactly match the exact decisions of the original paper, since the way they
            # extract layers is actually less precise than us (for each example they take the middle of the 12 outer blocks
            # of the densenet169 architecture, but the last 6 blocks have a lot more layers in them than the first 6.
            halfway_point = int((len(modules)+i) / 2)
            # modules[i].apply(add_hook)
            modules[halfway_point].apply(add_hook)

    def forward(self, x):
        _ = self.model(x)
        out = self.output
        self.output = []
        return out


def train(args, model, model_name, dataset_name, threat_model, eps):
    from attacks.bia.bia import TrainArgs
    args: TrainArgs
    if args.RN and args.DA:
        save_checkpoint_suffix = 'BIA+RN+DA'
    elif args.RN:
        save_checkpoint_suffix = 'BIA+RN'
    elif args.DA:
        save_checkpoint_suffix = 'BIA+DA'
    else:
        save_checkpoint_suffix = 'BIA'
    model.eval()
    model = Extractor(model, model_name).to(device).eval()
    print(f"Training eps: {eps}, dataset_name: {dataset_name} and model_name: {model_name}")
    # eps = epsilon[dataset_name]

    # we need to get the module ouput at a certain index for their technique, so we use the same method as
    # ssah to extract the features.
    # TODO: implement it.
    # Input dimensions
    if dataset_name == "imagenet":
        scale_size = 256
        img_size = 224
    elif dataset_name == "cifar10":
        scale_size = 32
        img_size = 32
    else:
        raise NotImplementedError
    netG = GeneratorResnet().to(device)
    model.set_encoder(torch.rand(1, 3, img_size, img_size).to(device))

    # Optimizer
    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    train_loader = load_data(dataset_name=dataset_name, threat_model=threat_model, model_name=model_name,
                            train=True, batch_size=args.batch_size, use_robust_5000=False, preprocessing=data_transform)
    if dataset_name == "imagenet":
        def default_normalize(t):
            t[:, 0, :, :] = (t[:, 0, :, :] - 0.485) / 0.229
            t[:, 1, :, :] = (t[:, 1, :, :] - 0.456) / 0.224
            t[:, 2, :, :] = (t[:, 2, :, :] - 0.406) / 0.225

            return t
    elif dataset_name == "cifar10":
        default_normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    else:
        raise NotImplementedError

    def normalize(t, mean, std):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean) / std
        t[:, 1, :, :] = (t[:, 1, :, :] - mean) / std
        t[:, 2, :, :] = (t[:, 2, :, :] - mean) / std

        return t


    # Loss
    criterion = nn.CrossEntropyLoss()  # idk why they have it in their code it doesn't do anything???
    bar = tqdm(range(args.epochs), total=(args.epochs*len(train_loader)))
    # Training
    for epoch in bar:
        running_loss = 0
        for i, (img, _) in enumerate(train_loader):
            img = img.to(device)
            netG.train()
            optimG.zero_grad()
            adv = netG(img)

            # Projection
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)

            if args.RN:
                mean = np.random.normal(0.50, 0.08)
                std = np.random.normal(0.75, 0.08)
                adv_out_slice = model(normalize(adv.clone(), mean, std))
                img_out_slice = model(normalize(img.clone(), mean, std))
            else:
                adv_out_slice = model(default_normalize(adv.clone()))
                img_out_slice = model(default_normalize(img.clone()))

            if args.DA:
                attention = [abs(torch.mean(img_out_slice_sub, dim=1, keepdim=True)).detach() for img_out_slice_sub in
                             img_out_slice]
            else:
                attention = [torch.ones(adv_out_slice_sub.shape).to(device) for adv_out_slice_sub in adv_out_slice]

            # we change it to the mean of the cosine similarities across the different models in case the model is an
            # ensemble. Behavior unchanged if it's a single model
            loss = sum([torch.cosine_similarity((adv_out_slice_sub*attention_sub).reshape(adv_out_slice_sub.shape[0], -1),
                                (img_out_slice_sub*attention_sub).reshape(img_out_slice_sub.shape[0], -1)).mean() for
                       (adv_out_slice_sub, img_out_slice_sub, attention_sub) in
                        zip(adv_out_slice, img_out_slice, attention)])/len(adv_out_slice)

            loss.backward()
            optimG.step()

            if i % 100 == 0:
            #     print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
                running_loss = 0
            running_loss += abs(loss.item())
            bar.set_description('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / ((i%100)+1)))
            bar.update(1)
            # One epoch
            # This will break if batch_size != 16 or we decide not to use imagenet. Since even with the batch size 16
            # on imagenet it's 80073 batches. Will swap it to just doing it after the epoch is done

            # if i % 80000 == 0 and i > 0:
            #     save_checkpoint_dir = os.path.join(bia_save_path, 'saved_models/{}'.format(model_name))
            #     if not os.path.exists(save_checkpoint_dir):
            #         os.makedirs(save_checkpoint_dir)
            #     save_path = os.path.join(save_checkpoint_dir, 'netG_{}_{}.pth'.format(save_checkpoint_suffix, epoch))
            #     torch.save(netG.state_dict(), save_path)
        save_checkpoint_dir = os.path.join(bia_save_path, 'saved_models/{}/{}'.format(dataset_name, model_name))
        if not os.path.exists(save_checkpoint_dir):
            os.makedirs(save_checkpoint_dir)
        save_path = os.path.join(save_checkpoint_dir, 'netG_{}_{}.pth'.format(save_checkpoint_suffix, epoch))
        torch.save(netG.state_dict(), save_path)
    return netG


def test():
    # import attacks.bia.model_layer.Vgg16_all_layer as Vgg16_all_layer
    # model = Vgg16_all_layer.Vgg16().to(device)
    # layer_idx = 16  # Maxpooling.3
    import attacks.bia.model_layer.Dense169_all_layer as Dense169_all_layer
    model = Dense169_all_layer.Dense169()
    layer_idx = 6  # Denseblock.2
    extractor = Extractor(model.model, "densenet").to(device)
    extractor.set_encoder(torch.rand((1, 3, 224, 224)))

    test_input = torch.rand((1, 3, 224, 224)).to(device)
    orig = model(test_input)[layer_idx]
    new = extractor(test_input)


if __name__ == '__main__':
    test()