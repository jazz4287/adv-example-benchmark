# adv-example-benchmark
Code for the paper "SoK: Analyzing Adversarial Examples: A Framework to Study Adversary Knowledge".

## Find our results
Our raw results can be found in the form of a json file "results.json" in the results folder. It is structured as follows:
 {Dataset:
   {Distinguisher norm:
    {Attack:
      {Target model:
        {experiment type: robust accuracy}}}}}
The experiments are as follows:
- ImageNet:
  - "robust_5000_adv_acc": attack with undefended surrogates
  - "worst_defended_Salman2020Do_R18_robust_5000_adv_acc": attack with worst-defended surrogate (Salman2020Do_R18)
  - "best_defended_Liu2023Comprehensive_Swin-L_robust_5000_adv_acc": attack with best-defended surrogate (Liu2023Comprehensive_Swin-L)
- CIFAR10:
  - "full_adv_acc": attack with undefended surrogates
  - "worst_defended_Ding2020MMA_full_adv_acc": attack with worst-defended surrogate (Ding2020MMA)
  - "best_defended_Peng2023Robust_full_adv_acc": attack with best-defended surrogate (Peng2023Robust)

We also provide some processed results in the notebooks/outputs/results.csv file. In, one can find the mean values of our metrics aggregated across all models for a given dataset, defense status, experiment, and attack. This table can be generated using the notebooks/paper_tables.ipynb notebook.

## Re-run our experiments
We provide two runs scripts in the scripts folder to run our experiments. The run_base scripts runs the experiments required to get the results for Table 2 in the paper. It runs the experiments with the undefended surrogates. We provide an example below.
Usage: ./run_base attack version cifar_def_bs cifar_und_bs imagenet_def_bs imagenet_und_bs combination

<code>./run_base Admix A 16 16 4 4 di-ti</code>

To evaluate the benign performance of the models, use the "benign" attack.
The version/variant of an attack describes whether to use the white-box or transferable version of an attack if it can. Below are the various versions available:
- AutoAttack:
  - no version = white-box
  - T = transferable
- ACG:
  - no version = white-box
- Admix, LGV, VNIFGSM:
  - A = transferable
  - B = white-box
-  BIA:
  - A = transferable
  - B = white-box
  - T-* = test versions, transferable
- Square: has no variants
- SSAH:
  - A: white-box
  - B: transferable
 
Some of the transferable attacks can use optional enhancements. We provide the list of attacks that can use them, with the name of the version that can use them and the name of the combination of enhancements:

- Admix:
  - A: ["di", "ti", "di-ti"]
- LGV:
  - A: ["di", "ti", "di-ti"]
- VNIFGSM:
    - A: ["di", "ti", "di-ti"]
- BIA:
  - A: ["da", "rn"]

The other script to run to get the other results we use is the script run_transfer_exp. It runs the transferable experiments using the worst and best defended surrogates and it has the following usage (bs stands for batch size), worst_cifar_bs stands for the batch size to use on CIFAR10 when using the worst-defended surrogate.

Usage: attack version worst_cifar_bs worst_imagenet_bs best_cifar_bs best_imagenet_bs combination


