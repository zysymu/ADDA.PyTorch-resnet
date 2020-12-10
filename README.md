# ADDA.PyTorch-resnet
Implementation of "Adversarial Discriminative Domain Adapation" in PyTorch

This repo is mostly based on https://github.com/Fujiki-Nakamura/ADDA.PyTorch

## Note
Before running the training code, make sure that `DATASETDIR` environment variable is set to dataset directory.

## Environment
- Python 3.8.5
- PyTorch 1.6.0

## Example
```
$ python train_source.py --logdir outputs
$ python main.py --logdir outputs --trained outputs/best_model.pt --slope 0.2
```

## Result
### SVHN -> MNIST
| | Paper | This Repo |
| --- | --- | --- |
| Source only | 0.601 | 0.659 |
| ADDA | 0.760 | ~0.83 |

## Resource
- https://arxiv.org/pdf/1702.05464.pdf
- https://github.com/Fujiki-Nakamura/ADDA.PyTorch
