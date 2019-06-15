# Tensorflow GANs Architectures Implementation

### Brief Description

GANs have proven to be very powerful generative models. So, here's a well-structured **Tensorflow** project containing implementations of some GANs architectures.

### Utilized Frameworks

- Tensorflow 1.13.1

### Repository Strucuture

**1) base folder:**
- contains abstract classes for both model and trainer.
**2) configs folder:**
- contains json files for different model configurations.
**3) data_loader folder:**
- contains data generator class for data loading and preprocessing.
**4) main folder:**
- contains main entry point for the project.
**5) models folder:**
- contains different model implementations.
**6) trainers folder:**
- contains trainers for models.
**7) utils folder:**
- contains logger for Tensorboard summary, argument parser, configuration processing and directory creation.

### Implemented Architectures

- DCGAN (Deep Convolutional Generative Adverserial Networks): https://arxiv.org/abs/1511.06434

### Usage

- Put your training images in data folder.
- Run the main file providing config and model arguments:
```
python main/main.py -c -m
```
- Have a nice day!

### To-Do List

1) Implement more GANs architectures.
2) Add Tensorflow 2.0 compatibility.
3) Add distributed training to the trainer process.
4) Improve the current training process and fix some issues.

### Acknowledgment

- https://github.com/MrGemy95/Tensorflow-Project-Template.git