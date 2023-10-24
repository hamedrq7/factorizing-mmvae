import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

parser.add_argument('--softmax', action='store_true', default=False,
                    help='apply softmax to scale of Laplace Distrubution')
parser.add_argument('--distr', type=str, default='Normal', metavar='M',
                    choices=['Laplace', 'Normal'],
                    help='distribution used for modeling prior, posterior and likelihood (default: Normal)')

args = parser.parse_args()

print(args)
args.softmax = True if args.distr == 'Laplace' else False
print(args)
