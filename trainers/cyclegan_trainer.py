from base.base_trainer import BaseTrain
from data_loader.data_generator import ImagePool
from tqdm import tqdm
import numpy as np
from scipy.misc import imsave

class CycleGANTrain(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(CycleGANTrain, self).__init__(sess, model, data, config, logger)
        self.lr_decay_step = self.config.learning_rate / 100
        self.pool = ImagePool()

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        g_loss_values = []
        d_loss_values = []
        for _ in loop:
            g_loss, d_loss = self.train_step()
            g_loss_values.append(g_loss)
            d_loss_values.append(d_loss)
            self.sess.run(self.model.increment_global_step_tensor)
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
        lr = self.config.learning_rate
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        if  cur_epoch > 100:
            lr = lr - (cur_epoch - 100) * self.lr_decay_step       
        t_batch_a = self.data[0].next_batch().eval()
        t_batch_b = self.data[1].next_batch().eval()
        g_feed_dict = {self.model.real_A : t_batch_a, self.model.real_B : t_batch_b, self.model.lr : lr}
        fake_A, fake_B, _, gen_loss = self.sess.run([self.model.fake_A, self.model.fake_B, 
                                    self.model.g_optim, self.model.g_loss], feed_dict=g_feed_dict)
        [fake_A, fake_B] = self.pool([fake_A, fake_B])
        d_feed_dict = {self.model.real_A : t_batch_a, self.model.real_B : t_batch_b, self.model.fake_A_sample : fake_A,
                    self.model.fake_B_sample : fake_B, self.model.lr : lr}
        _, dis_loss = self.sess.run([self.model.d_optim, self.model.d_loss], feed_dict=d_feed_dict)  
        if (self.model.global_step_tensor.eval(self.sess) % 100 == 0):
            self.show_results(t_batch_a, t_batch_b)                          
        return gen_loss, dis_loss

    def show_results(self, t_batch_a, t_batch_b):
        feed_dict = {self.model.real_A : t_batch_a, self.model.real_B : t_batch_b}
        fake_A, fake_B = self.sess.run([self.model.fake_A, self.model.fake_B], feed_dict=feed_dict)
        for i in range(fake_A.shape[0]):
            imsave("results/result_epoch_{}_A".format(self.model.cur_epoch_tensor.eval(self.sess)), fake_A[i,:,:,:])
            imsave("results/result_epoch_{}_B".format(self.model.cur_epoch_tensor.eval(self.sess)), fake_B[i,:,:,:])