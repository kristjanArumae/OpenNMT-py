import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 'True', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch Size')

    parser.add_argument('--epochs',
                        type=int,
                        default=32,
                        help='Num max epochs')

    parser.add_argument('--unchanged_limit',
                        type=int,
                        default=10,
                        help='Early stopping limit')

    parser.add_argument('--sent_len',
                        type=int,
                        default=45,
                        help='Max inp len')

    parser.add_argument('--train',
                        type='bool',
                        default=True,
                        help='Training Mode')

    parser.add_argument("--balance",
                        type=float,
                        default=None,
                        help="If used, then it becomes the negative sentence class downsample")

    parser.add_argument('--bert_model',
                        type=str,
                        default='bert-base-uncase',
                        help='pretrained BERT model')

    parser.add_argument("--lr",
                        type=float,
                        default=1e-5,
                        help="Learning rate")

    parser.add_argument('--optim',
                        type=str,
                        default='adam',
                        help='Optimizer to use')

    parser.add_argument("--weights",
                        type=float,
                        default=None,
                        help="If used, then it becomes the negative sentence class weight during training")

    return parser.parse_args()
