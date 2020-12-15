import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--model', type=str, default='default')
    # train
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='outputs/garbage')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='amazon')
    parser.add_argument('--tgt_cat', type=str, default='webcam')
    parser.add_argument('--message', '-m',  type=str, default='')
    args, unknown = parser.parse_known_args()
    if args.model == 'default':
        from core.experiment import run
    elif args.model == 'resnet50':
        from core.experiment_rn50 import run
    run(args)
