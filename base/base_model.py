import tensorflow as tf

class BaseModel:
    """An abstract class definition for any implemented model"""
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        """Saves the current parameters of the model"""
        print("Saving Model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model Saved!")

    def load(self, sess):
        """Loads the last checkpoint of the model parameters"""
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading Model Checkpoint {} ....\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded!")

    def init_cur_epoch(self):
        """Initializes the epoch count of the training process"""
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """Initializes the global step count of the training process"""
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self):
        raise NotImplementedError   

    def build_model(self):
        raise NotImplementedError                               

        

