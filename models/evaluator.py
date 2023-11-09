import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm, get_voc_pallete, colorize_mask, get_image_list
import utils


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.palette = get_voc_pallete(2)
        self.is_eval = args.is_eval
        if args.is_eval:
            self.image_list = get_image_list(Dataset_Path=os.path.join("F:\\Dataset_RS\\ChangeDetect", args.data_name + "_256"), split="test")
            self.output_dir = args.output_dir
        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if True:
        # if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            # checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            # print(self.net_G)
            checkpoint = torch.load(r"G:\2\CD_VIG_V20_2_WHU_b8_lr0.0001_adamw_train_val_200_linear_ce_multi_train_False_multi_infer_False_shuffle_AB_False_embed_dim_256_4\best_ckpt.pt", map_location=self.device)
            # print(checkpoint['model_G_state_dict'])
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            # self.net_G = torch.load("H:\BIT权重\CD_base_transformer_pos_s4_dd8_WHU_b8_lr0.01_sgd_train_val_200_linear_ce_multi_train_False_multi_infer_False_shuffle_AB_False_embed_dim_256\\84", map_location=self.device)
            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        assert self.args.n_class == G_pred.shape[1]
        # G_pred = torch.argmax(G_pred, dim=1)
        if self.args.n_class > 1:
            G_pred = torch.argmax(G_pred, dim=1)
        elif self.args.n_class == 1:
            G_pred = (G_pred >= 0.5).long()

            G_pred = torch.squeeze(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        # 仿照simeCD中可视化代码
        if self.is_eval:
            prediction_im = colorize_mask(torch.squeeze(G_pred).cpu().numpy(), self.palette)
            save_path = os.path.join(self.output_dir, self.image_list[self.batch_id].split("/")[-1].split(".")[0] + ".png")
            prediction_im.save(save_path)
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # 可视化，在eval时被自己注释了
        # if np.mod(self.batch_id, 100) == 1:
        #     vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        #     vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        #
        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())
        #
        #     vis_gt = utils.make_numpy_grid(self.batch['L'])
        #     vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #     file_name = os.path.join(
        #         self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        #     plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

    def eval_models(self, checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
