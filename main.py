import tensorflow as tf

from data_loader.data_generator import DataGenerator
from utils.arg_parser import get_args
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger

def main():

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(-1)

    sess = tf.Session()

    if (args.model == 'dcgan'):
        from models.dcgan_model import DCGANModel
        from trainers.dcgan_trainer import DCGANTrain

        create_dirs([config.summary_dir, config.checkpoint_dir, config.resized_data_dir])
        data = DataGenerator(config.data_dir, config.resized_data_dir, config.t_size, config.batch_size)
        model = DCGANModel(config)
        logger = Logger(sess, config)
        trainer = DCGANTrain(sess, model, data, config, logger)
        model.load(sess)
        trainer.train()

    elif (args.model == 'cyclegan'):
        from models.cyclegan_model import CycleGANModel
        from trainers.cyclegan_trainer import CycleGANTrain

        create_dirs([config.summary_dir, config.checkpoint_dir, config.resized_data_dir_a, config.resized_data_dir_b])
        dataA = DataGenerator(config.data_dir_a, config.resized_data_dir_a, config.t_size, config.batch_size)
        dataB = DataGenerator(config.data_dir_b, config.resized_data_dir_b, config.t_size, config.batch_size)
        model = CycleGANModel(config)
        logger = Logger(sess, config)
        trainer = CycleGANTrain(sess, model, [dataA, dataB], config, logger)
        model.load(sess)
        trainer.train()

    else:
        print("model doesn't exist")
        exit(-1)    


if __name__ == '__main__':
    main()
