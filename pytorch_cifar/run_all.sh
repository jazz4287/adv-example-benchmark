#!/bin/bash


MODEL_NAMES=("vgg" "resnet18" "preactresnet18" "googlenet" "densenet121" "resnext29" "mobilenet" "mobilenetv2" "dpn92" "shufflenetg2" "senet18" "shufflenetv2" "efficientnet-b0" "regnetx_200mf" "simpledla") #"resnet50")

for model_name in "${MODEL_NAMES[@]}"
do
  echo "###############################"
  echo "Training model: ${model_name}"
  echo "###############################"
  python main.py --model "${model_name}"
done
