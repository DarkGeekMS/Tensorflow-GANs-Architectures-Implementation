import argparse

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-m', '--model',
        metavar='M',
        default='dcgan',
        help='The GAN Model to be trained'
    )    
    args = argparser.parse_args()
    return args
