from base.base_trainer import BaseTrain
from tqdm import tqdm
import numpy as np
from scipy.misc import imsave

class DCGANTrain(BaseTrain):
    """A class for th trainer of DCGAN model where training logic is executed"""
    def __init__(self, sess, model, data, config, logger):
        super(DCGANTrain, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        """Executes the logic of a single epoch"""
        loop = tqdm(range(self.config.num_iter_per_epoch))
        g_loss_values = []
        d_loss_values = []
        # looping over the number of step per epoch and running optimizers and getting losses
        for _ in loop:
            g_loss, d_loss = self.train_step()
            g_loss_values.append(g_loss)
            d_loss_values.append(d_loss)
            self.sess.run(self.model.increment_global_step_tensor)
        total_g_loss = np.mean(g_loss_values)
        total_d_loss = np.mean(d_loss_values)

        # writing summaries and saving the current checkpoint
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'generator_loss' : total_g_loss,
            'discriminator_loss' : total_d_loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        # sampling results at the end of each epoch  
        self.show_results()

    def train_step(self):
        """Executes the logic of a single training step"""
        t_batch = self.data.next_batch().eval()
        feed_dict = {self.model.train_data : t_batch}
        _, gen_loss, dis_loss = self.sess.run([self.model.train_step, self.model.g, self.model.d], feed_dict=feed_dict)
        return gen_loss, dis_loss       

    def show_results(self):
        """Samples output from the generator and saving them"""
        images = self.model.sample_images()
        self.model.load(self.sess)    
        generated = sess.run(images)
        imsave("results/result_epoch_{}".format(self.model.cur_epoch_tensor.eval(self.sess)), generated)
