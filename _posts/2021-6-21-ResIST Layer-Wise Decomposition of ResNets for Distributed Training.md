---
layout: post
title: ResIST&#58; Layer-Wise Decomposition of ResNets for Distributed Training
---


## Abstract
We propose **ResIST**, a novel distributed training protocol for Residual Networks (ResNets). **ResIST** randomly decomposes a global ResNet into several shallow sub-ResNets that are trained independently in a distributed manner for several local iterations, before having their updates synchronized and aggregated into the global model. In the next round, new sub-ResNets are randomly generated and the process repeats. By construction, per iteration, **ResIST** communicates only a small portion of network parameters to each machine and never uses the full model during training. Thus, **ResIST** reduces the communication, memory, and time requirements of ResNet training to only a fraction of the requirements of previous methods. In comparison to common protocols like data-parallel training and data-parallel training with local SGD, **ResIST** yields a decrease in wall-clock training time, while being competitive with respect to model performance.

{% include image.html url="/images/resist/resnet_ist.png" description="Figure 1: The ResIST model: we depict the process of partitioning the layers of a ResNet to different sub-ResNets, then aggregating the updated parameters back into the global network. Row (a) represents the original global ResNet. Row (b) shows the creation of two sub-ResNets. Observe that subnetwork 1 contains the residual blocks 1, 2 and 4, while subnetwork 2 contains the residual blocks 3, 4 and 5. Row (c) shows the reassembly of the global ResNet, after locally training subnetworks 1 and 2 for some number of local SGD iterations; residual blocks that are common across subnetworks (e.g., residual block 4) are aggregated appropriately during the reassembly." %}

## Introduction
In recent years, the field of Computer Vision (CV) has seen a revolution, beginning with the introduction of AlexNet during the ILSVRC2012 competition. 
Following this initial application of deep convolutional neural networks (CNNs), more modern architectures were produced, thus rapidly pushing the state of the art in image recognition. In particular, the introduction of the residual connection (ResNets) allowed these networks to be scaled to massive depths without being crippled by issues of unstable gradients during training. Such ability to train large networks was only furthered by the development of architectural advancements, like batch normalization. The capabilities of ResNets have been further expanded in recent years, but the basic ResNet architecture has remained widely-used. While ResNets have become a standard building block for the advancement of CV research, the computational requirements for training them are significant. For example, training a ResNet50 on ImageNet with a single NVIDIA M40 GPU takes 14 days. 

Therefore, distributed training with multiple GPUs is commonly adopted to speed up the training process for ResNets. Yet, such acceleration is achieved at the cost of a remarkably large number of GPUs (e.g 256 NVIDIA Tesla P100 GPU). Additionally, frequent synchronization and high communication costs create bottlenecks that hinder such methods from achieving linear speedup with respect to the number of available GPUs. Asynchronous approaches avoid the cost of synchronization, but stale updates complicate their optimization process. Other methods, such as data-parallel training with local SGD, reduce the frequency of synchronization. Similarly, model-parallel training has gained in popularity by decreasing the cost of local training between synchronization rounds.

### This Project
We focus on efficient distributed training of convolutional neural networks with residual skip connections. Our proposed methodology accelerates synchronous, distributed training by leveraging ResNet robustness to layer removal. In particular, a group of high-performing subnetworks (sub-ResNets) is created by partitioning the layers of a shared ResNet model to create multiple, shallower sub-ResNets. These sub-ResNets are then trained independently (in parallel) for several iterations before aggregating their updates into the global model and beginning the next iteration. Through the local, independent training of shallow sub-ResNets, this methodology both limits synchronization and communicates fewer parameters per synchronization cycle, thus drastically reducing communication overhead. We name this scheme **ResNet Independent Subnetwork Training** (**ResIST**).

The outcome of this work can be summarized as follows:
1. We propose a distributed training scheme for ResNets, dubbed **ResIST**, that partitions the layers of a global model to multiple, shallow sub-ResNets, which are then trained independently between synchronization rounds.
2. We perform extensive ablation experiments to motivate the design choices for **ResIST**, indicating that optimal performance is achieved by i) using pre-activation ResNets, ii) scaling intermediate activations of the global network at inference time, iii) sharing layers between sub-ResNets that are sensitive to pruning, and iv) imposing a minimum depth on sub-ResNets during training.
3. **ResIST** is shown to achieve high accuracy and time efficiency in all cases. We conduct experiments on several image classification and object detection datasets, including CIFAR10/100, ImageNet, and PascalVOC.
4. We utilize **ResIST** to train numerous different ResNet architectures (e.g., ResNet101, ResNet152, and ResNet200) and provide implementations for each in PyTorch

## Methods

**ResIST** operates by partitioning the layers of a global ResNet to different, shallower sub-ResNets, training those independently, and intermittently aggregating their updates into the global model. The high-level process followed by **ResIST** is depicted in Fig 1 and outlined in more detail by pusedocode below.

### Model Architecture
To achieve optimal performance with **ResIST**, the global model must be sufficiently deep.
Otherwise, sub-ResNets may become too shallow after partitioning, leading to poor performance.
For most experiments, a ResNet101 architecture is selected, which balances sufficient depth with reasonable computational complexity.

**ResIST** performs best with pre-activation ResNets. Intuitively, applying batch normalization prior to the convolution ensures that the input distribution of remaining residual blocks will remain fixed, even when certain layers are removed from the architecture.

{% include image.html url="/images/resist/resnet_model.png" description="Figure 2: The pre-activation ResNet101 model used in the majority of experiments. The figure identifies the convolutional blocks that are partitioned to subnetworks. The network is comprised of four major sections, each containing a certain number of convolutional blocks of equal channel dimension." %}

### Sub-ResNet Construction

Pruning literature has shown that strided layers, initial layers, and final layers within CNNs are sensitive to pruning. Additionally, repeated blocks of identical convolutions (i.e., equal channel size and spatial resolution) are less sensitive to pruning. Drawing upon these results, as shwon in Figure 2, **ResIST** only partitions blocks within the third section of the ResNet, while all other blocks are shared between sub-ResNets.

These blocks are chosen for partitioning because i) they account for the majority of network layers; ii) they are not strided; iii) they are located within the middle of the network (i.e., initial and final layers are excluded); and iv) they reside within a long chain of identical convolutions.

By partitioning only these blocks, **ResIST** allows sub-ResNets to be shallower than the global network, while maintaining high performance.

The process of constructing sub-ResNets follows a simple procedure, depicted in Fig 1.
As shown in the transition from row (a) to (b) within Fig. 1, indices of partitioned layers within the global model are randomly permuted and distributed to sub-ResNets in a round-robin fashion. Each sub-ResNet receives an equal number of convolutional blocks (e.g., see row (b) within Fig. 1). In certain cases, residual blocks may be simultaneously partitioned to multiple sub-ResNets to ensure sufficient depth (e.g., see block 4 in Fig. 1.

**ResIST** produces subnetworks with 1/S of the global model depth, where S represents the number of independently-trained sub-ResNets.

The shallow sub-ResNets created by **ResIST** accelerate training and reduce communication in comparison to methods that communicate and train the full model.
Table 1 shows the comparison of local SGD to **ResIST** with respect to the amount of data communicated during each synchronization round for different numbers of machines, highlighting the superior communication-efficiency of **ResIST**.

{% include image.html url="/images/resist/table_1.png" description="Table 1: Reports the amount of data communicated during each communication round (in GB) of both local SGD and ResIST across different numbers of machines with ResNet101." %}
