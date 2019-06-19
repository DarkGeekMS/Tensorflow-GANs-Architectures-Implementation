import argparse

def get_args():
    """Parses argument from terminal,
    Two arguments:
    - config: path to the configuration file
    - model: the name of the model to be trained"""
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
