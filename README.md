
### Introduction

+ ***./data*** provides the ChaLearn2013 data substet.
+ ***./src*** provides the implementation of our work. Within them,
  + ***./src/model/*** is the implementation of our proposed network,
  + ***the others*** is the official implementation of our compared methods.
+ ***./checkpoints*** provides the pre-trained network parameters.
+ ***./configs*** provides the network and training configurations corresponding to the pre-trained models in ./checkpoints .
+ ***./tools*** provides the detailed training, validation, and testing procedure we used.
+ ***./train.sh*** is the bash file for training. You can select different experiment configuration and train your own model by modifying this file.
+ ***./test.sh*** is the bash file for evaluation. 

### Requirements

+ torch = 1.7.1
+ torchvision = 0.8.2
+ signatory = 1.2.3.1.6.0

### Usage

### Training

```
sh train.sh
```

### Evaluation

```
sh test.sh
```





