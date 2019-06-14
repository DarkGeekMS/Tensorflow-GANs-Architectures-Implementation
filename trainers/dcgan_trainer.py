from base.base_trainer import BaseTrain
from tqdm import tqdm
import numpy as np

class DCGANTrain(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(DCGANTrain, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        g_loss_values = []
        d_loss_values = []
        for _ in loop:
            g_loss, d_loss = self.train_step()
            g_loss_values.append(g_loss)
            d_loss_values.append(d_loss)
        total_g_loss = np.mean(g_loss_values)
        total_d_loss = np.mean(d_loss_values)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'generator_loss' : total_g_loss,
            'discriminator_loss' : total_d_loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        t_batch = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.train_data : t_batch}
        _, gen_loss, dis_loss = self.sess.run([self.model.train_step, self.model.g, self.model.d], feed_dict=feed_dict)
        return gen_loss, dis_loss       