from base.base_trainer import BaseTrain
from data_loader.data_generator import ImagePool
from tqdm import tqdm
import numpy as np
from scipy.misc import imsave

class CycleGANTrain(BaseTrain):
    """A class for th trainer of CycleGAN model where training logic is executed"""
    def __init__(self, sess, model, data, config, logger):
        super(CycleGANTrain, self).__init__(sess, model, data, config, logger)
        self.lr_decay_step = self.config.learning_rate / 100
        self.pool = ImagePool()

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
        cur_it = self.model.global_step_tensor.eval(session=self.sess)
        summaries_dict = {
            'generator_loss' : total_g_loss,
            'discriminator_loss' : total_d_loss
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)    

    def train_step(self):
        """Executes the logic of a single training step"""
        # getting learning rate (constant for 100 epochs, then decays linearly to zero)
        lr = self.config.learning_rate
        cur_epoch = self.model.cur_epoch_tensor.eval(session=self.sess)
        if  cur_epoch > 100:
            lr = lr - (cur_epoch - 100) * self.lr_decay_step 
        # getting next batch of data from both domains          
        t_batch_a = self.data[0].next_batch().eval(session=self.sess)
        t_batch_b = self.data[1].next_batch().eval(session=self.sess)
        # running generator optimizer and getting losses and fake outputs
        g_feed_dict = {self.model.real_A : t_batch_a, self.model.real_B : t_batch_b, self.model.lr : lr}
        fake_A, fake_B, _, gen_loss = self.sess.run([self.model.fake_A, self.model.fake_B, 
                                    self.model.g_optim, self.model.g_loss], feed_dict=g_feed_dict)
        # creating a pool of 50 fake outputs                            
        [fake_A, fake_B] = self.pool([fake_A, fake_B])
        # running discriminator optimizer and getting losses
        d_feed_dict = {self.model.real_A : t_batch_a, self.model.real_B : t_batch_b, self.model.fake_A_sample : fake_A,
                    self.model.fake_B_sample : fake_B, self.model.lr : lr}
        _, dis_loss = self.sess.run([self.model.d_optim, self.model.d_loss], feed_dict=d_feed_dict)
        # sampling results at the end of each epoch  
        if (self.model.global_step_tensor.eval(session=self.sess) % 100 == 0):
            self.show_results(t_batch_a, t_batch_b)                          
        return gen_loss, dis_loss

    def show_results(self, t_batch_a, t_batch_b):
        """Samples output from the generator and saving them"""
        feed_dict = {self.model.real_A : t_batch_a, self.model.real_B : t_batch_b}
        fake_A, fake_B = self.sess.run([self.model.fake_A, self.model.fake_B], feed_dict=feed_dict)
        for i in range(fake_A.shape[0]):
            imsave("results/result_epoch_{}_A".format(self.model.cur_epoch_tensor.eval(session=self.sess)), fake_A[i,:,:,:])
            imsave("results/result_epoch_{}_B".format(self.model.cur_epoch_tensor.eval(session=self.sess)), fake_B[i,:,:,:])