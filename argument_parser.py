import argparse

def default_parser():
    parser = argparse.ArgumentParser(description="Distillation")
    parser.add_argument('--task', nargs='?', type=str, default='DistilBert',
                        help='Set distillation model type')
    parser.add_argument('--input_dir', nargs='?', type=str, default='dataset/',
                        help='')
    parser.add_argument('--teacher_dir', nargs='?', type=str, default='models/',
                        help='Knowledge learned from teacher model')
    parser.add_argument("--T", default=10., type=float,
                        help="Temperature for distillation")
    return parser.parse_args()



