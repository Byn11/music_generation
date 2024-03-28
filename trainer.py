import torch
import torch.nn as nn
from collections import defaultdict
import random
import pickle
from scipy.sparse import csc_matrix

from utils import logger
from utils.metrics import cal_metrics,ac_loss
from utils.utils1 import *
from dataset import collate_fn

x_fontdict = {'fontsize': 6,
             'verticalalignment': 'top',
             'horizontalalignment': 'left',
             'rotation': 'vertical',
             'rotation_mode': 'anchor'}
y_fontdict = {'fontsize': 6}



class BaseTrainer:
    def __init__(self, asset_path, model, criterion, optimizer,ac_optimizer,
                 train_loader, eval_loader, test_loader, device,
                 config):
        self.asset_path = asset_path
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.ac_optimizer=ac_optimizer
        self.device = device
        self.config = config
        self.verbose = config['verbose']

        # metrics
        self.metrics = config['metrics']

        # dataloader
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.loading_epoch = 1
        self.current_step = 0
        self.losses = defaultdict(list)

    def _step(self, loss, **kwargs):
        raise NotImplementedError()

    def _epoch(self, epoch, mode, **kwargs):
        raise NotImplementedError()

    def train(self, **kwargs):
        raise NotImplementedError()

    def load_model(self, restore_epoch, load_rhythm=False):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        restore_ckpt = os.path.join(self.asset_path, 'model', 'checkpoint_%d.pth.tar' % restore_epoch)
        if not (os.path.isfile(restore_ckpt) or load_rhythm):
            logger.info("no checkpoint with %d epoch" % restore_epoch)
        else:
            if os.path.isfile(restore_ckpt):
                checkpoint = torch.load(restore_ckpt, map_location=self.device)
            else:
                rhythm_asset_path = os.path.join('/'.join(self.asset_path.split('/')[:-1]),
                                                 'idx%03d' % self.config['restore_rhythm']['idx'])
                rhythm_ckpt = os.path.join(rhythm_asset_path, 'model',
                                           'checkpoint_%d.pth.tar' % self.config['restore_rhythm']['epoch'])
                checkpoint = torch.load(rhythm_ckpt, map_location=self.device)
            if load_rhythm:
                model_dict = model.state_dict()
                rhythm_state_dict = {k: v for k, v in checkpoint['model'].items() if 'rhythm' in k}
                model_dict.update(rhythm_state_dict)
                model.load_state_dict(model_dict)
                logger.info("restore rhythm model")
            else:
                model.load_state_dict(checkpoint['model'])
                #assert isinstance(self.optimizer.load_state_dict, object)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.current_step = checkpoint['current_step']
                self.loading_epoch = checkpoint['epoch'] + 1
                logger.info("restore model with %d epoch" % restore_epoch)

    def save_model(self, epoch, current_step):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        logger.info('saving model, Epoch %d, step %d' % (epoch, current_step))
        model_save_path = os.path.join(self.asset_path, 'model', 'checkpoint_%d.pth.tar' % epoch)
        state_dict = {'model': model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'current_step': current_step,
                      'epoch': epoch}
        torch.save(state_dict, model_save_path)

    def adjust_learning_rate(self, factor=.5, min_lr=0.000001, min_aclr=0.000001):
        losses = self.losses['eval']
        if len(losses) > 4 and losses[-1] > np.mean(losses[-4:-1]):
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * factor, min_lr)
                param_group['lr'] = new_lr
                logger.info('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))
            for i, param_group in enumerate(self.ac_optimizer.param_groups):
                old_aclr = float(param_group['ac_lr'])
                new_aclr = max(old_aclr * factor, min_aclr)
                param_group['ac_lr'] = new_aclr
                logger.info('adjusting ac learning rate from %.6f to %.6f' % (old_aclr, new_aclr))

class CMTtrainer(BaseTrainer):
    def __init__(self, asset_path, model, criterion, optimizer,ac_optimizer,
                 train_loader, eval_loader, test_loader,
                 device, config):
        super(CMTtrainer, self).__init__(asset_path, model, criterion, optimizer,ac_optimizer,
                                         train_loader, eval_loader, test_loader,
                                         device, config)
        # for logging
        self.losses = defaultdict(list)
        self.asset_path=asset_path
        self.tf_writer = get_tfwriter(asset_path)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax=nn.Softmax(dim=-1)

    def train(self, **kwargs):
        # load model if exists
        self.load_model(kwargs["restore_epoch"], kwargs["load_rhythm"])

        # start training
        for epoch in range(self.loading_epoch, self.config['max_epoch']):
            logger.info("")
            logger.info("%d epoch" % epoch)

            #train epoch
            #-----------------------------------------------------------
            logger.info("==========train %d epoch==========" % epoch)
            self.ac_epoch(epoch, 'train', self.config['rhythm_only'])
            # valid epoch and sampling
            with torch.no_grad():
                logger.info("==========valid %d epoch==========" % epoch)
                self.ac_epoch(epoch, 'eval', self.config['rhythm_only'])
                if epoch > self.loading_epoch and ((epoch < 150 and epoch % 10== 0) or epoch % 150 == 0):
                    self.save_model(epoch, self.current_step)
                if not self.config['rhythm_only']:
                    self._sampling(epoch)
            #------------------------------------------------------------

        # #------------------TF train 完了--------------------
        # logger.info("==========RL start==========" )
        # for i in range(100):
        #     logger.info("==========epoch%d=========="%i)
        #     self.ac()


        self.tf_writer.close()

    def ac_step(self,loss,**kwargs):
        self.ac_optimizer.zero_grad()

        loss.backward()
        self.ac_optimizer.step()
        print('更新成功')


    def _step(self, loss, **kwargs):
        # back-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.current_step += 1

    def _epoch(self, epoch, mode, rhythm_only=False, **kwargs):
        # enable eval mode
        if mode == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.eval_loader

        results = defaultdict(float)
        total_pitch_loss = 0.
        total_rhythm_loss = 0.
        total_pitch1_loss = 0.
        total_rhythm1_loss = 0.
        num_total = 0
        for i, data in enumerate(loader):
            # preprocessing and forwarding
            for key in data.keys():
                data[key] = data[key].to(self.device)
            result_dict = self.model(data['rhythm'], data['pitch'],
                                     data['chord'], data['rhythm1'], data['pitch1'],False, rhythm_only)
            rhythm_out = result_dict['rhythm']
            rhythm_out = self.log_softmax(rhythm_out)
            rhythm_out = rhythm_out.view(-1, rhythm_out.size(-1))
            num_total += rhythm_out[:, 0].numel()
            if not rhythm_only:
                pitch_out = result_dict['pitch']  #(16,128,50)
                pitch_out = self.log_softmax(pitch_out)
                pitch_out = pitch_out.view(-1, pitch_out.size(-1)) #(2048,50)
                rhythm1_out = result_dict['rhythm1'] #(16,128,3)
                rhythm1_out = self.log_softmax(rhythm1_out)
                rhythm1_out = rhythm1_out.view(-1, rhythm1_out.size(-1)) #(2048,3)
                pitch1_out = result_dict['pitch1'] #(16,128,48)
                pitch1_out = self.log_softmax(pitch1_out)
                pitch1_out = pitch1_out.view(-1, pitch1_out.size(-1)) #(2048,48)

            # get loss & metric(accuracy)
            rhythm_criterion = self.criterion[0]
            pitch_criterion = self.criterion[1]
            rhythm1_criterion = self.criterion[0]
            pitch1_criterion = self.criterion[1]#多音版本为[2]

            rhythm_loss = rhythm_criterion(rhythm_out, data['rhythm'][:, 1:].contiguous().view(-1))
            total_rhythm_loss += rhythm_loss.item()

            result = dict()
            result.update(cal_metrics(rhythm_out, data['rhythm'][:, 1:].contiguous().view(-1),
                                      self.metrics, mode, name='rhythm'))

            if rhythm_only:
                pitch_loss = 0
            else:
                pitch_loss = pitch_criterion(pitch_out, data['pitch'][:, 1:].contiguous().view(-1))
                total_pitch_loss += pitch_loss.item()
                result.update(cal_metrics(pitch_out, data['pitch'][:, 1:].contiguous().view(-1),
                                          self.metrics, mode, name='pitch'))

                rhythm1_loss = rhythm1_criterion(rhythm1_out, data['rhythm1'][:, 1:].contiguous().view(-1))
                total_rhythm1_loss += rhythm1_loss.item()
                result.update(cal_metrics(rhythm1_out, data['rhythm1'][:, 1:].contiguous().view(-1),
                                          self.metrics, mode, name='rhythm1'))

                pitch1_loss = pitch1_criterion(pitch1_out, data['pitch1'][:, 1:].contiguous().view(-1))
                total_pitch1_loss += pitch1_loss.item()
                result.update(cal_metrics(pitch1_out, data['pitch1'][:, 1:].contiguous().view(-1),
                                         self.metrics, mode, name='pitch1'))
            loss = pitch_loss + rhythm_loss+ pitch1_loss + rhythm1_loss

            for key, val in result.items():
                results[key] += val

            # do training operations
            if mode == 'train':
                self._step(loss)
                # self._step(rhythm_loss)
                if self.verbose and self.current_step % 100 == 0:
                    logger.info("%d training steps" % self.current_step)
                    print_dict = {'nll': loss.item()}
                    print_dict.update({
                        'nll_pitch': pitch_loss,
                        'nll_rhythm': rhythm_loss,
                        'nll_pitch1': pitch1_loss,
                        'nll_rhythm1': rhythm1_loss
                    })
                    print_result(print_dict, result)

        # logging epoch statistics and information
        results = {key: val / len(loader) for key, val in results.items()}
        footer = '/' + mode
        losses = {'nll' + footer: (total_rhythm_loss + total_pitch_loss+total_rhythm1_loss + total_pitch1_loss) / len(loader),
                  'nll_pitch' + footer: total_pitch_loss / len(loader),
                  'nll_rhythm' + footer: total_rhythm_loss / len(loader),
                  'nll_pitch1' + footer: total_pitch1_loss / len(loader),
                  'nll_rhythm1' + footer: total_rhythm1_loss / len(loader)
                  }
        print_result(losses, results)
        #self.tf_writer = get_tfwriter(self.asset_path)
        tensorboard_logging_result(self.tf_writer, epoch, losses)
        tensorboard_logging_result(self.tf_writer, epoch, results)
        #self.tf_writer.close()

        self.losses[mode].append((total_rhythm_loss + total_pitch_loss) / len(loader))
        if mode == 'eval':
            self.adjust_learning_rate()

    def ac_epoch(self, epoch, mode, rhythm_only=False, ac_l=False,**kwargs):
        # enable eval mode
        if mode == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.eval_loader

        results = defaultdict(float)
        total_pitch_loss = 0.
        total_rhythm_loss = 0.
        total_pitch1_loss = 0.
        total_rhythm1_loss = 0.
        num_total = 0
        for i, data in enumerate(loader):
            # preprocessing and forwarding
            for key in data.keys():
                data[key] = data[key].to(self.device)
            result_dict = self.model(data['rhythm'], data['pitch'],#(16,129)
                                     data['chord'], data['rhythm1'], data['pitch1'],False, rhythm_only)
            rhythm_out = result_dict['rhythm']
            ac_rhythm_out = self.log_softmax(rhythm_out)

            rhythm_out = self.log_softmax(rhythm_out)
            rhythm_out = rhythm_out.view(-1, rhythm_out.size(-1))
            num_total += rhythm_out[:, 0].numel()
            if not rhythm_only:
                pitch_out = result_dict['pitch']  #(16,128,50)
                ac_pitch_out = self.softmax(pitch_out)
                pitch_out = self.log_softmax(pitch_out)
                pitch_out = pitch_out.view(-1, pitch_out.size(-1)) #(2048,50)

                rhythm1_out = result_dict['rhythm1'] #(16,128,3)
                ac_rhythm1_out = self.log_softmax(rhythm1_out)
                rhythm1_out = self.log_softmax(rhythm1_out)
                rhythm1_out = rhythm1_out.view(-1, rhythm1_out.size(-1)) #(2048,3)

                pitch1_out = result_dict['pitch1'] #(16,128,50)
                ac_pitch1_out = self.softmax(pitch1_out)
                pitch1_out = self.log_softmax(pitch1_out)
                pitch1_out = pitch1_out.view(-1, pitch1_out.size(-1))  # (2048,50)


            # get loss & metric(accuracy)
            rhythm_criterion = self.criterion[0]
            pitch_criterion = self.criterion[1]
            rhythm1_criterion = self.criterion[0]
            pitch1_criterion = self.criterion[1]#多音版本为[2]

            rhythm_loss = rhythm_criterion(rhythm_out, data['rhythm'][:, 1:].contiguous().view(-1))
            total_rhythm_loss += rhythm_loss.item()

            result = dict()
            result.update(cal_metrics(rhythm_out, data['rhythm'][:, 1:].contiguous().view(-1),
                                      self.metrics, mode, name='rhythm'))

            if rhythm_only:
                pitch_loss = 0
            else:
                pitch_loss = pitch_criterion(pitch_out, data['pitch'][:, 1:].contiguous().view(-1))
                total_pitch_loss += pitch_loss.item()
                result.update(cal_metrics(pitch_out, data['pitch'][:, 1:].contiguous().view(-1),
                                          self.metrics, mode, name='pitch'))

                rhythm1_loss = rhythm1_criterion(rhythm1_out, data['rhythm1'][:, 1:].contiguous().view(-1))
                total_rhythm1_loss += rhythm1_loss.item()
                result.update(cal_metrics(rhythm1_out, data['rhythm1'][:, 1:].contiguous().view(-1),
                                          self.metrics, mode, name='rhythm1'))

                pitch1_loss = pitch1_criterion(pitch1_out, data['pitch1'][:, 1:].contiguous().view(-1))
                total_pitch1_loss += pitch1_loss.item()
                result.update(cal_metrics(pitch1_out, data['pitch1'][:, 1:].contiguous().view(-1),
                                        self.metrics, mode, name='pitch1'))
            loss = pitch_loss + rhythm_loss+ pitch1_loss + rhythm1_loss
            p1_acloss ,p_acloss= 0,0

            for j in range(ac_pitch1_out.size()[1]):#pitch1_out=(16,128,50)
                r_list1=get_reward1(data['pitch'][:, 1:],#(16,128)
                                  data['rhythm1'][:, 1:],
                                  data['pitch1'][:, 1:],
                                  data['chord_prime'][:,1:,:],
                                  j)
                r_list1 = torch.tensor(r_list1).to(self.device)
                temp1 = ac_pitch1_out[:,j,:]   # pitch_out=(16,128,50) temp=[16,1,pitch_num]

                r_list2=get_reward2(data['rhythm'][:,1:],
                                    data['pitch'][:, 1:],#(16,129)
                                    data['chord_prime'][:, 1:,:],#(12,16,129)
                                    j)
                r_list2 = torch.tensor(r_list2).to(self.device)
                temp2 = ac_pitch_out[:,j, :]


                pitch1_acloss=-ac_loss(temp1,r_list1) #r_list1 (50,16)   temp1 (16,50)
                pitch_acloss=-ac_loss(temp2,r_list2)
                p1_acloss=p1_acloss+pitch1_acloss
                p_acloss=p_acloss+pitch_acloss
            #损失函数求完，可以进行一次学习
            p1_acloss=p1_acloss/len(ac_pitch1_out)/100
            p_acloss = p_acloss / len(ac_pitch1_out) / 100
            #print('第',i,'次','  p_acloss=',p_acloss,'  p1_acloss=',p1_acloss,'  loss=',loss)
            if mode == 'train':
                if ac_l==False:
                    self._step(loss)
                    ac_l=True
                else:
                    self.ac_step(p1_acloss+p_acloss)
                    ac_l=False

            for key, val in result.items():
                results[key] += val

        #logging epoch statistics and information
        results = {key: val / len(loader) for key, val in results.items()}
        footer = '/' + mode
        losses = {'nll' + footer: (total_rhythm_loss + total_pitch_loss+total_rhythm1_loss + total_pitch1_loss) / len(loader),
                  'nll_pitch' + footer: total_pitch_loss / len(loader),
                  'nll_rhythm' + footer: total_rhythm_loss / len(loader),
                  'nll_pitch1' + footer: total_pitch1_loss / len(loader),
                  'nll_rhythm1' + footer: total_rhythm1_loss / len(loader)
                  }
        print_result(losses, results)
        #self.tf_writer = get_tfwriter(self.asset_path)
        tensorboard_logging_result(self.tf_writer, epoch, losses)
        tensorboard_logging_result(self.tf_writer, epoch, results)
        #self.tf_writer.close()

        self.losses[mode].append((total_rhythm_loss + total_pitch_loss) / len(loader))
        if mode == 'eval':
            self.adjust_learning_rate()


    def _sampling(self, epoch):
        self.load_model(90, False)
        self.model.eval()
        loader = self.test_loader #正确为test
        asset_path = os.path.join(self.asset_path)

        indices = random.sample(range(len(loader.dataset)), self.config["num_sample"])
        batch = collate_fn([loader.dataset[i] for i in indices])
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        prime = batch['pitch'][:, :self.config["num_prime"]]
        prime_rhythm1=batch['rhythm1'][:,:self.config['num_prime']]
        prime1 = batch['pitch1'][:, :self.config['num_prime']]
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        prime_rhythm = batch['rhythm'][:, :self.config["num_prime"]]
        result_dict = model.sampling(prime_rhythm, prime, prime_rhythm1, prime1,batch['chord'],
                                     self.config["topk"], self.config['attention_map'])
        result_key = 'pitch'
        result_key1='pitch1'
        pitch_idx = result_dict[result_key].cpu().numpy()
        pitch1_idx = result_dict[result_key1].cpu().numpy()

        logger.info("==========sampling result of epoch %03d==========" % epoch)
        os.makedirs(os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % epoch), exist_ok=True)

        for sample_id in range(pitch_idx.shape[0]):
            logger.info(("Sample %02d : " % sample_id) + str(pitch_idx[sample_id][self.config["num_prime"]:self.config["num_prime"]+20]))
            save_path = os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % epoch,
                                     'epoch%03d_sample%02d.mid' % (epoch, sample_id))
            gt_pitch = batch['pitch'].cpu().numpy()
            gt_chord = batch['chord'][:, :-1].cpu().numpy()
            gt_pitch1 = batch['pitch1'].cpu().numpy()

            sample_dict = {'pitch': pitch_idx[sample_id],
                           'rhythm': result_dict['rhythm'][sample_id].cpu().numpy(),
                           'pitch1': pitch1_idx[sample_id],
                           'rhythm1': result_dict['rhythm1'][sample_id].cpu().numpy(),
                           'chord': csc_matrix(gt_chord[sample_id])}


            with open(save_path.replace('.mid', '.pkl'), 'wb') as f_samp:
                pickle.dump(sample_dict, f_samp)
            instruments = pitch_to_midi(pitch_idx[sample_id], pitch1_idx[sample_id],gt_chord[sample_id], model.frame_per_bar, save_path)
            save_instruments_as_image(save_path.replace('.mid', '.jpg'), instruments,
                                      frame_per_bar=model.frame_per_bar,
                                      num_bars=(model.max_len // model.frame_per_bar))

            # save groundtruth
            logger.info(("Groundtruth %02d : " % sample_id) +
                        str(gt_pitch[sample_id, self.config["num_prime"]:self.config["num_prime"] + 20]))
            gt_path = os.path.join(asset_path, 'sampling_results', 'epoch_%03d' % epoch,
                                     'epoch%03d_groundtruth%02d.mid' % (epoch, sample_id))
            gt_dict = {'pitch': gt_pitch[sample_id, :-1],
                       'rhythm': batch['rhythm'][sample_id, :-1].cpu().numpy(),
                       'pitch1': gt_pitch1[sample_id, :-1],
                       'rhythm1': batch['rhythm1'][sample_id, :-1].cpu().numpy(),
                       'chord': csc_matrix(gt_chord[sample_id])}
            with open(gt_path.replace('.mid', '.pkl'), 'wb') as f_gt:
                pickle.dump(gt_dict, f_gt)
            gt_instruments = pitch_to_midi(gt_pitch[sample_id, :-1], gt_pitch1[sample_id, :-1],gt_chord[sample_id], model.frame_per_bar, gt_path)
            save_instruments_as_image(gt_path.replace('.mid', '.jpg'), gt_instruments,
                                      frame_per_bar=model.frame_per_bar,
                                      num_bars=(model.max_len // model.frame_per_bar))

            if self.config['attention_map']:
                os.makedirs(os.path.join(asset_path, 'attention_map', 'epoch_%03d' % epoch,
                                         'RDec-Chord', 'sample_%02d' % sample_id), exist_ok=True)

                for head_num in range(8):
                    for l, w in enumerate(result_dict['weights_bdec']):
                        fig_w = plt.figure(figsize=(8, 8))
                        ax_w = fig_w.add_subplot(1, 1, 1)
                        heatmap_w = ax_w.pcolor(w[sample_id, head_num].cpu().numpy(), cmap='Reds')
                        ax_w.set_xticks(np.arange(0, self.model.module.max_len))
                        ax_w.xaxis.tick_top()
                        ax_w.set_yticks(np.arange(0, self.model.module.max_len))
                        ax_w.set_xticklabels(rhythm_to_symbol_list(result_dict['rhythm'][sample_id].cpu().numpy()),
                                             fontdict=x_fontdict)
                        chord_symbol_list = [''] * pitch_idx.shape[1]
                        for t in sorted(chord_array_to_dict(gt_chord[sample_id]).keys()):
                            chord_symbol_list[t] = chord_array_to_dict(gt_chord[sample_id])[t].tolist()
                        ax_w.set_yticklabels(chord_to_symbol_list(gt_chord[sample_id]), fontdict=y_fontdict)
                        ax_w.invert_yaxis()
                        plt.savefig(os.path.join(asset_path, 'attention_map', 'epoch_%03d' % epoch, 'RDec-Chord',
                                                 'sample_%02d' % sample_id,
                                                 'epoch%03d_RDec-Chord_sample%02d_head%02d_layer%02d.jpg' % (
                                                 epoch, sample_id, head_num, l)))
                        plt.close()

