import tensorflow as tf
import time

class BaseTrain:
    """An abstract class definition for the trainers of any implemented model"""
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        """Loops over epoch count and calls the function executing the training logic"""
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(session=self.sess), self.config.num_epochs + 1, 1):
            start_time = time.time()
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            print("Epoch: {}, Time: {}".format(cur_epoch, time.time() - start_time))

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError            
