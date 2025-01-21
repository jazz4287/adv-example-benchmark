from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack
import torch.nn.functional as F
import numpy as np
from scipy import stats as st


class Admix(Attack):
    """The Admix attack.

    From the paper 'Admix: Enhancing the Transferability of Adversarial Attacks',
    https://arxiv.org/abs/2102.00436

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        portion: Portion for the mixed image. Defaults to 0.2.
        size: Number of randomly sampled images. Defaults to 3.
        num_classes: Number of classes of the dataset used. Defaults to 1001.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        portion: float = 0.2,
        size: int = 3,
        num_classes: int = 1000,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
        di: bool = False,
        ti: bool = False
    ) -> None:
        super().__init__(normalize, device)

        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.portion = portion
        self.size = size
        self.num_classes = num_classes
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()
        self.kernel_name = "gaussian"
        self.len_kernel = 7
        self.nsig = 3
        self.resize_rate = 0.9
        self.diversity_prob = 0.5
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.di = di
        self.ti = ti

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform Admix on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        x_adv = x.clone().detach()
        stacked_kernel = self.stacked_kernel.to(self.device)

        scales = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Admix + MI-FGSM
        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            # Add delta to original image then admix
            x_admix = self.admix(x_adv)
            x_admixs = torch.cat([x_admix * scale for scale in scales])

            # Compute loss
            if self.di:
                outs = self.model(self.normalize(self.input_diversity(x_admixs)))
            else:
                outs = self.model(self.normalize(x_admixs))

            # One-hot encode labels for all admixed images
            one_hot = nn.functional.one_hot(y, self.num_classes)
            one_hot = torch.cat([one_hot] * 5 * self.size).float()

            loss = self.lossfn(outs, one_hot)

            if self.targeted:
                loss = -loss

            # Gradients
            grad = torch.autograd.grad(loss, x_admixs)[0]
            if self.ti:
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)

            # Split gradients and compute mean
            split_grads = torch.tensor_split(grad, 5, dim=0)
            grads = [g * s for g, s in zip(split_grads, scales, strict=True)]
            grad = torch.mean(torch.stack(grads), dim=0)

            # Gather gradients
            split_grads = torch.tensor_split(grad, self.size)
            grad = torch.sum(torch.stack(split_grads), dim=0)

            # Apply momentum term
            g = self.decay * g + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )

            # Update perturbed image
            x_adv = x_adv.detach() + self.alpha * g.sign()
            x_adv = x + torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        return x_adv

    def admix(self, x: torch.Tensor) -> torch.Tensor:
        def x_admix(x: torch.Tensor) -> torch.Tensor:
            return x + self.portion * x[torch.randperm(x.shape[0])]

        return torch.cat([(x_admix(x)) for _ in range(self.size)])

    def kernel_generation(self):
        if self.kernel_name == "gaussian":
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == "linear":
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == "uniform":
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(
            np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
            / (kernlen + 1)
            * 2
        )
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x


if __name__ == '__main__':
    from torchattack.runner import run_attack

    run_attack(Admix, {'eps': 8 / 255, 'steps': 10})
