# ADDA.PyTorch-resnet
Implementation of "Adversarial Discriminative Domain Adaptation" in PyTorch.

This repo is mostly based on https://github.com/Fujiki-Nakamura/ADDA.PyTorch

## Note
Before running the training code, make sure that `DATASETDIR` environment variable is set to dataset directory.

## Environment
- Python 3.8.5
- PyTorch 1.6.0

## Example
For training on SVHN-MNIST
```
$ python train_source.py --logdir outputs
$ python main.py --logdir outputs --trained outputs/best_model.pt --slope 0.2
```

For training on Office dataset using ResNet-50
```
$ python core/train_source_rn50.py --n_classes 31 --lr 1e-5 --src_cat amazon --tgt_cat webcam
$ python main.py --n_classes 31 --trained outputs/garbage/best_model.pt --lr 1e-5 --d_lr 1e-4 --logdir outputs --model resnet50 --src-cat amazon --tgt-cat webcam
```

## Result
### SVHN -> MNIST
| | Paper | This Repo |
| --- | --- | --- |
| Source only | 0.601 | 0.646 |
| ADDA | 0.760 | 0.805 |

### Office-31 Amazon -> Office-31 Webcam
| | Paper | This Repo |
| --- | --- | --- |
| Source only | 0.684 | 0.714 |
| ADDA | 0.862 | 0.831 |

## Resources
- https://arxiv.org/pdf/1702.05464.pdf
- https://github.com/Fujiki-Nakamura/ADDA.PyTorch
- https://github.com/erictzeng/adda/issues/11
- https://github.com/corenel/pytorch-adda/issues/15
