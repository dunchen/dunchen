---
layout: post
title: ResIST\: Layer-Wise Decomposition of ResNets for Distributed Training
---

## Abstract
We propose **ResIST**, a novel distributed training protocol for Residual Networks (ResNets). **ResIST** randomly decomposes a global ResNet into several shallow sub-ResNets that are trained independently in a distributed manner for several local iterations, before having their updates synchronized and aggregated into the global model. In the next round, new sub-ResNets are randomly generated and the process repeats. By construction, per iteration, **ResIST** communicates only a small portion of network parameters to each machine and never uses the full model during training. Thus, **ResIST** reduces the communication, memory, and time requirements of ResNet training to only a fraction of the requirements of previous methods. In comparison to common protocols like data-parallel training and data-parallel training with local SGD, **ResIST** yields a decrease in wall-clock training time, while being competitive with respect to model performance.

{% include image.html url="/images/resist/resnet_ist.png" description="here is the caption" %}

## Introduction
In recent years, the field of Computer Vision (CV) has seen a revolution, beginning with the introduction of AlexNet during the ILSVRC2012 competition \cite{alexnet, imagenet}. 
Following this initial application of deep convolutional neural networks (CNNs), more modern architectures were produced, thus rapidly pushing the state of the art in image recognition \cite{zfnet, googlenet, vgg}. 
In particular, the introduction of the residual connection (ResNets) allowed these networks to be scaled to massive depths without being crippled by issues of unstable gradients during training \cite{resnet}. 
Such ability to train large networks was only furthered by the development of architectural advancements, like batch normalization \cite{batchnorm}. 
The capabilities of ResNets have been further expanded in recent years, but the basic ResNet architecture has remained widely-used \cite{resnext, preactres}.
While ResNets have become a standard building block for the advancement of CV research \cite{fasterrcnn, densenets, maskrcnn, retinanet}, the computational requirements for training them are significant. For example, training a ResNet50 on ImageNet with a single NVIDIA M40 GPU takes 14 days. \cite{you2018imagenet} 