import argparse
import os
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import tensorboard
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.dataset import *
from copy import deepcopy
from torch.backends import cudnn
from data.dataset import WHU_Dataset
from segmentation_models_pytorch.decoders.unet.model import *
import torch.multiprocessing
import warnings
import cv2

import models.DSIFN as IFN
import models.SNUNet as SNUNet
import models.SiamUnet_conc as SiamUnet_conc
import models.SiamUnet_diff as SiamUnet_diff
import models.Unet as Unet
import models.DTCDSCN as DTCDSCN

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--root_path", type=str, default="/home/chrisd/change/STCD/data/", help="root path")
parser.add_argument("--dataset_name", type=str, default="LEVIR", help="name of the dataset")
parser.add_argument("--CDdataset_name", type=str, default="LEVIR", help="name of the dataset")
parser.add_argument("--save_name", type=str, default="runs/STCD", help="experiments name")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--load_path", type=str, default="/home/chrisd/change/STCD/resume/", help="load path")
args = parser.parse_args()
print(args)


def main():

    print(cudnn.is_available())
    torch.backends.cudnn.benchmark = True

    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # model, optimizer = init_basic_elems()
    model, optimizer = init_seg_cd_net()
    # print the model
    load_model = False
    if load_model:
        load_different_models = False
        if load_different_models:
            models = []
            resume1 = os.path.join(args.load_path, '20.00_model.pth')
            resume2 = os.path.join(args.load_path, '40.00_model.pth')
            resume3 = os.path.join(args.load_path, '60.00_model.pth')
            if os.path.isfile(resume1):
                print("=> loading checkpoint model1 '{}'".format(resume1))
                model.module.load_state_dict(torch.load(resume1))
                models.append(deepcopy(model))
                print("=> loading checkpoint model2 '{}'".format(resume2))
                model.module.load_state_dict(torch.load(resume2))
                models.append(deepcopy(model))
                print("=> loading checkpoint model3 '{}'".format(resume3))
                model.module.load_state_dict(torch.load(resume3))
                models.append(deepcopy(model))
            else:
                print("=> no checkpoint found at resume")
                print("=> Will start from scratch.")
        else:
            resume = os.path.join(args.load_path, 'WHU-pseudo-change.pth')
            print("=> loading checkpoint best_model'{}'".format(resume))
            model.module.load_state_dict(torch.load(resume))

        select_data = False
        if select_data:
            cd_loader = DataLoader(
                CD_Dataset(root_path=args.root_path, dataset=args.CDdataset_name, train_val='train'),
                batch_size=1, shuffle=False,
                pin_memory=False, num_workers=args.n_cpu, drop_last=False)
            for i in range(len(models)):
                models[i].eval()
            tbar = tqdm(cd_loader)
            id_to_reliability = []
            metric = SegmentationMetric(numClass=2, device='cuda:0')
            with torch.no_grad():
                for i, (image_A, image_B, label, image_name) in enumerate(tbar):
                    image_A, image_B, label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                              label.cuda(non_blocking=True).unsqueeze(1)
                    preds = []
                    for model in models:
                        seg_A, seg_B, diffseg = model(image_A, image_B)
                        change_prediction = torch.sigmoid(diffseg)
                        change_prediction = (change_prediction > 0.5).int()
                        preds.append(change_prediction.cpu())

                    mIOU = []
                    for i in range(len(preds) - 1):
                        metric.addBatch(preds[i], preds[-1])

                        mIOU.append(metric.IntersectionOverUnion()[1])

                    reliability = sum(mIOU) / len(mIOU)
                    id_to_reliability.append((image_name, reliability))

                id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
                with open(os.path.join(args.root_path, args.CDdataset_name, 'train', 'list', "reliable_ids.txt"),
                          'w') as f:
                    for elem in id_to_reliability[:len(id_to_reliability) // 2]:
                        f.write(elem[0][0] + '\n')
                with open(os.path.join(args.root_path, args.CDdataset_name, 'train', 'list', "unreliable_ids.txt"),
                          'w') as f:
                    for elem in id_to_reliability[len(id_to_reliability) // 2:]:
                        f.write(elem[0][0] + '\n')

        generate_label = True
        if generate_label:

            cd_loader = DataLoader(
            CD_Dataset(root_path=args.root_path, dataset=args.CDdataset_name, train_val='train', reliable='lunreliable'),
            batch_size=1, shuffle=False,
            pin_memory=False, num_workers=1, drop_last=False)
            cd_total = SegmentationMetric(numClass=2, device='cuda:0')
            model.eval()
            tbar = tqdm(cd_loader)
            with torch.no_grad():
                for i, (image_A, image_B, label, image_name) in enumerate(tbar):
                    image_A, image_B, label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                              label.cuda(non_blocking=True).unsqueeze(1)
                    seg_A, seg_B, diffseg = model(image_A, image_B)
                    # diffseg = model(image_A, image_B)
                    # cd_pre = torch.sigmoid(torch.min(cd_map, diffseg))

                    change_prediction = torch.sigmoid(diffseg)

                    vis_feature = False
                    if vis_feature:
                        x_visualize = change_prediction[0][0].cpu().numpy()  # 用Numpy处理返回的[1,256,513,513]特征图
                        x_visualize = (((x_visualize - np.min(x_visualize)) /
                                        (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
                        savedir = '/home/chrisd/change/STCD/data/val_pred_temp/'
                        x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
                        cv2.imwrite(savedir + image_name[0], x_visualize)

                    change_prediction = (change_prediction > 0.5).int()
                    # aaa = change_prediction.squeeze(0).squeeze(0).cpu().numpy()
                    # sum_a = aaa.sum()

                    # pred_A = torch.sigmoid(seg_A)
                    # pred_A = (pred_A > 0.5).int()  # N C H W
                    # pred_B = torch.sigmoid(seg_B)
                    # pred_B = (pred_B > 0.5).int()  # N C H W
                    # change_prediction_seg = torch.abs(pred_B - pred_A)

                    # cd_all = torch.min(change_prediction_seg, change_prediction)
                    cd_total.addBatch(change_prediction.cpu(), label.cpu())
                    # f1 = cd_total.F1score()[1]
                    # iou = cd_total.IntersectionOverUnion()[1]
                    # tbar.set_description('f1: %.2f iou: %.2f' % (f1, iou))

                    change_prediction[change_prediction == 1] = 255
                    pred_cd = Image.fromarray(np.uint8(change_prediction.squeeze(0).squeeze(0).cpu().numpy()))
                    # pred_cd.show()

                    # pred_img = cv2.imread(os.path.join(args.root_path, args.CDdataset_name, 'train', 'pseudo_label_or', image_name[0]))
                    # pred_img = cv2.morphologyEx(pred_img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                    # cv2.imwrite(os.path.join(args.root_path, args.CDdataset_name, 'train', 'pseudo_label', image_name[0]), pred_img)

                    # change_prediction_seg[change_prediction_seg == 1] = 255
                    # pred_cd_seg = Image.fromarray(change_prediction_seg.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8), mode='L')
                    # pred_cd_seg.show()

                    # pred_cd.save(os.path.join(args.root_path, args.CDdataset_name, 'train', 'pseudo_label_B', image_name[0]))

                    pred_cd.save(os.path.join(args.root_path, args.CDdataset_name, 'train', 'pseudo_label', image_name[0]))

            cd_f1 = cd_total.F1score()[1]
            cd_iou = cd_total.IntersectionOverUnion()[1]
            cd_OA = cd_total.OverallAccuracy()
            cd_Pre = cd_total.Precision()[1]
            cd_Rec = cd_total.Recall()[1]
            print('change predictions:  train f1 %.3f, iou: %.3f, OA %.3f, Pre: %.3f, Rec: %.3f'
                  % (cd_f1, cd_iou, cd_OA, cd_Pre, cd_Rec))

    # trainloader = DataLoader(
    #     SC_Dataset(root_path=args.root_path, dataset=args.dataset_name, train_val='train', semi=True),
    #     batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.n_cpu, drop_last=True)

    trainloader = DataLoader(
        LEVIR_Dataset(root_path=args.root_path, dataset=args.dataset_name, train_val='train'),
        batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.n_cpu, drop_last=True)

    # trainloader = DataLoader(
    #     FFC_Dataset(root_path=args.root_path, dataset=args.dataset_name, train_val='train'),
    #     batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.n_cpu, drop_last=True)

    val_dataloader = DataLoader(
        CD_Dataset(root_path=args.root_path, dataset=args.CDdataset_name, train_val='val'),
        batch_size=4, shuffle=False, num_workers=4)

    criterion = BCE_DICE()
    # best_model = train_ffctl(model, trainloader, val_dataloader, optimizer, criterion, args)
    best_model = train_semi_cd(model, trainloader, val_dataloader, optimizer, criterion, args)

    print("The ending")


def train_ffctl(model, trainloader, valloader, optimizer, criterion, args):
    iters = 0
    previous_best = 0.0

    iter_per_epoch = len(trainloader)
    lr_scheduler = Poly(optimizer=optimizer, num_epochs=args.n_epochs, iters_per_epoch=iter_per_epoch)

    writer_dir = os.path.join(args.save_name)
    writer = tensorboard.SummaryWriter(writer_dir)

    for epoch in range(1, args.n_epochs + 1):
        train_acc = SegmentationMetric(numClass=2, device='cuda:0')
        print("\n==> Epoch %i, previous best = %.2f" % (epoch, previous_best))

        model.train()
        wrt_mode = 'train'
        total_cd_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (image_A, image_B, cd_label) in enumerate(tbar):
            image_A, image_B, cd_label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                         cd_label.cuda(non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()

            seg_A, seg_B, diff = model(image_A, image_B)
            # diff = model(image_A, image_B)

            change_prediction = torch.sigmoid(diff)
            cd_loss = criterion(change_prediction, cd_label.float())

            change_prediction = (change_prediction > 0.5).int()  # N C H W
            train_acc.addBatch(change_prediction.cpu(), cd_label.cpu())
            trian_f1 = train_acc.F1score()[1]
            train_cd_iou = train_acc.IntersectionOverUnion()[1]

            writer.add_scalar(f'{wrt_mode}/trian_f1', trian_f1, iters)
            writer.add_scalar(f'{wrt_mode}/train_cd_iou', train_cd_iou, iters)

            loss_all = cd_loss
            loss_all.backward()

            optimizer.step()
            total_cd_loss += cd_loss.item()

            iters += 1
            lr_scheduler.step(epoch=epoch - 1)

            tbar.set_description('CD_Loss: %.3f, train_cd_iou: %.3f' %
                                 ((total_cd_loss / (i + 1)), train_cd_iou))
            writer.add_scalar(f'{wrt_mode}/CD_Loss', total_cd_loss / (i + 1), iters)

        model.eval()

        wrt_mode = 'val'
        tbar = tqdm(valloader)

        total_val_loss_cd = 0.0
        with torch.no_grad():
            cd_acc = SegmentationMetric(numClass=2, device='cuda:0')
            for i, (image_A, image_B, cd_label, image_name) in enumerate(tbar):
                image_A, image_B, cd_label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                             cd_label.cuda(non_blocking=True).unsqueeze(1)
                seg_A, seg_B, diff = model(image_A, image_B)
                # diff = model(image_A, image_B)

                change_prediction = torch.sigmoid(diff)

                val_cd_loss = criterion(change_prediction, cd_label.float())
                total_val_loss_cd += val_cd_loss

                change_prediction = (change_prediction > 0.5).int()  # N C H W
                cd_acc.addBatch(change_prediction.cpu(), cd_label.cpu())

                # tbar.set_description(
                #     'cd_f1: %.2f cd_iou: %.2f ' % (cd_f1, cd_iou))
                writer.add_scalar(f'{wrt_mode}/cd_val_Loss', total_val_loss_cd / (i + 1), iters)

            cd_f1 = cd_acc.F1score()[1]
            cd_iou = cd_acc.IntersectionOverUnion()[1]
            cd_OA = cd_acc.OverallAccuracy()
            cd_Pre = cd_acc.Precision()[1]
            cd_Rec = cd_acc.Recall()[1]

            writer.add_scalar(f'{wrt_mode}/cd_f1', cd_f1, epoch)
            writer.add_scalar(f'{wrt_mode}/cd_iou', cd_iou, epoch)
            writer.add_scalar(f'{wrt_mode}/cd_OA', cd_OA, epoch)
            writer.add_scalar(f'{wrt_mode}/cd_Pre', cd_Pre, epoch)
            writer.add_scalar(f'{wrt_mode}/cd_Rec', cd_Rec, epoch)

            print('epoch %d, train cd_f1 %.3f, cd_iou: %.3f' % (epoch, cd_f1, cd_iou))

        if cd_iou > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_name, '%.2f_best_model.pth' % previous_best))
            previous_best = cd_iou
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_name, '%.2f_best_model.pth' % cd_iou))  # ema_model
            best_model = deepcopy(model)

        if epoch in [args.n_epochs // 3, args.n_epochs * 2 // 3, args.n_epochs]:
            torch.save(model.module.state_dict(), os.path.join(args.save_name, '%.2f_model.pth' % epoch))

    return best_model


def contrastive_loss(pred, cd_label, pse_label, img_name):

    cd_pred = pred[:cd_label.shape[0]]
    pse_pred = pred[cd_label.shape[0]:]

    # for i in range(4):
    #     x_visualize = cd_pred[i][0].detach().cpu().numpy()  # 用Numpy处理返回的[1,256,513,513]特征图
    #     x_visualize = (((x_visualize - np.min(x_visualize)) /
    #                     (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
    #     savedir = '/home/chrisd/change/STCD/vis_ctloss/true_change/'
    #     x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
    #     cv2.imwrite(savedir + img_name[i], x_visualize)
    #
    #
    #     x_visualize = pse_pred[i][0].detach().cpu().numpy()  # 用Numpy处理返回的[1,256,513,513]特征图
    #     x_visualize = (((x_visualize - np.min(x_visualize)) /
    #                     (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
    #     savedir = '/home/chrisd/change/STCD/vis_ctloss/pse_change/'
    #     x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
    #     cv2.imwrite(savedir + img_name[i], x_visualize)


    M = (torch.zeros(cd_label.size())).cuda()
    M[cd_label == pse_label] = 1

    N = (torch.zeros(cd_label.size())).cuda()
    N[cd_label != pse_label] = 1

    # cd_or = cd_pred.cpu().detach().numpy()
    # cd_pred[N == 1] = torch.abs(cd_pred[N == 1] - 1)
    # cd_num = cd_pred.cpu().detach().numpy()

    neg_cdpre = torch.abs(cd_pred - 1)

    # cd_label[cd_label == 1] = 255
    # cd_label = Image.fromarray(np.uint8(cd_label[0].squeeze(0).cpu().numpy()))
    # cd_label.show()
    # M[M == 1] = 255
    # pred_cd = Image.fromarray(np.uint8(M[0].squeeze(0).cpu().numpy()))
    # pred_cd.show()
    # pse_label[pse_label == 1] = 255
    # pse_label = Image.fromarray(np.uint8(pse_label[0].squeeze(0).cpu().numpy()))
    # pse_label.show()

    loss_pos = F.mse_loss(pse_pred, cd_pred, reduction='none') * M
    loss_pos = loss_pos.sum() / (M.sum() + 1e-8)

    loss_neg = F.mse_loss(pse_pred, neg_cdpre, reduction='none') * N
    loss_neg = loss_neg.sum() / (N.sum() + 1e-8)

    # loss_ct = F.mse_loss(pse_pred, cd_pred, reduction='mean')
    return loss_pos + loss_neg


def train_semi_cd(model, trainloader, valloader, optimizer, criterion, args):
    iters = 0
    previous_best = 0.0

    cd_ce_loss = torch.nn.BCELoss(reduction='mean')

    iter_per_epoch = len(trainloader)
    lr_scheduler = Poly(optimizer=optimizer, num_epochs=args.n_epochs, iters_per_epoch=iter_per_epoch)

    writer_dir = os.path.join(args.save_name)
    writer = tensorboard.SummaryWriter(writer_dir)

    for epoch in range(1, args.n_epochs + 1):
        print("\n==> Epoch %i, previous best = %.2f" % (epoch, previous_best))
        train_acc = SegmentationMetric(numClass=2, device='cuda:0')
        model.train()
        wrt_mode = 'train'
        total_loss = 0.0
        total_cd_loss = 0.0
        total_ct_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (image_A, image_B, s_label_A, s_label_B, cd_label, CA, CB, CL, img_name) in enumerate(tbar):

            image_A, image_B, s_label_A, s_label_B, cd_label = image_A.cuda(non_blocking=True), \
                                                               image_B.cuda(non_blocking=True), \
                                                               s_label_A.cuda(non_blocking=True).unsqueeze(1), \
                                                               s_label_B.cuda(non_blocking=True).unsqueeze(1), \
                                                               cd_label.cuda(non_blocking=True).unsqueeze(1)
            CA, CB, CL = CA.cuda(non_blocking=True), CB.cuda(non_blocking=True), \
                         CL.cuda(non_blocking=True).unsqueeze(1),
            optimizer.zero_grad()

            data_A = torch.cat((image_A, CA), dim=0)
            data_B = torch.cat((image_B, CB), dim=0)
            c_label_data = torch.cat((cd_label, CL), dim=0)

            seg_A, seg_B, diff = model(data_A, data_B)
            seg_A = torch.sigmoid(seg_A)
            seg_loss_A = criterion(seg_A[:image_A.shape[0]], s_label_A.float())
            # seg_B = torch.sigmoid(seg_B)
            # seg_loss_B = cd_ce_loss(seg_B[:image_B.shape[0]], s_label_B.float())

            change_prediction = torch.sigmoid(diff)
            cd_loss = criterion(change_prediction, c_label_data.float())
            # cd_loss = cd_ce_loss(change_prediction, c_label_data.float())
            ct_loss = contrastive_loss(change_prediction, cd_label, CL, img_name)

            change_prediction = (change_prediction > 0.5).int()   # N C H W
            train_acc.addBatch(change_prediction.cpu(), c_label_data.cpu())
            trian_f1 = train_acc.F1score()[1]
            train_cd_iou = train_acc.IntersectionOverUnion()[1]

            writer.add_scalar(f'{wrt_mode}/trian_f1', trian_f1, iters)
            writer.add_scalar(f'{wrt_mode}/train_cd_iou', train_cd_iou, iters)

            # loss_all = cd_loss + ct_loss
            loss_all = seg_loss_A + cd_loss + ct_loss     # + seg_loss_B
            # loss_all = seg_loss_A + seg_loss_B + cd_loss + ct_loss
            # loss_all = cd_loss
            loss_all.backward()

            optimizer.step()
            total_loss += seg_loss_A.item()
            total_cd_loss += cd_loss.item()
            total_ct_loss += ct_loss.item()

            iters += 1
            lr_scheduler.step(epoch=epoch - 1)

            tbar.set_description('Seg_Loss: %.3f,  CD_Loss: %.3f, CT_loss: %.3f' %
                                 (total_loss / (i + 1), total_cd_loss / (i + 1), total_ct_loss / (i + 1)))
            writer.add_scalar(f'{wrt_mode}/Seg_Loss', total_loss / (i + 1), iters)
            writer.add_scalar(f'{wrt_mode}/CD_Loss', total_cd_loss / (i + 1), iters)
            writer.add_scalar(f'{wrt_mode}/CT_Loss', total_ct_loss / (i + 1), iters)

        model.eval()

        wrt_mode = 'val'
        tbar = tqdm(valloader)

        total_val_loss = 0.0
        total_val_loss_cd = 0.0
        with torch.no_grad():
            cd_acc = SegmentationMetric(numClass=2, device='cuda:0')
            for i, (image_A, image_B, cd_label, image_name) in enumerate(tbar):
                image_A, image_B, cd_label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                             cd_label.cuda(non_blocking=True).unsqueeze(1)
                seg_A, seg_B, diff = model(image_A, image_B)
                change_prediction = torch.sigmoid(diff)

                val_cd_loss = criterion(change_prediction, cd_label.float())
                # val_cd_loss = cd_ce_loss(change_prediction, cd_label.float())
                total_val_loss_cd += val_cd_loss

                change_prediction = (change_prediction > 0.5).int()  # N C H W
                cd_acc.addBatch(change_prediction.cpu(), cd_label.cpu())

                writer.add_scalar(f'{wrt_mode}/cd_val_Loss', total_val_loss_cd / (i + 1), iters)

        cd_f1 = cd_acc.F1score()[1]
        cd_iou = cd_acc.IntersectionOverUnion()[1]
        cd_OA = cd_acc.OverallAccuracy()
        cd_Pre = cd_acc.Precision()[1]
        cd_Rec = cd_acc.Recall()[1]

        print('epoch %d, train cd_f1 %.3f, cd_iou: %.3f' % (epoch, cd_f1, cd_iou))

        writer.add_scalar(f'{wrt_mode}/cd_f1', cd_f1, epoch)
        writer.add_scalar(f'{wrt_mode}/cd_iou', cd_iou, epoch)
        writer.add_scalar(f'{wrt_mode}/cd_OA', cd_OA, epoch)
        writer.add_scalar(f'{wrt_mode}/cd_Pre', cd_Pre, epoch)
        writer.add_scalar(f'{wrt_mode}/cd_Rec', cd_Rec, epoch)

        if cd_iou > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_name, '%.2f_best_model.pth' % previous_best))
            previous_best = cd_iou
            torch.save(model.module.state_dict(), os.path.join(args.save_name, '%.2f_best_model.pth' % cd_iou))  #  ema_model
            best_model = deepcopy(model)

        if epoch in [args.n_epochs // 3, args.n_epochs * 2 // 3, args.n_epochs]:
            torch.save(model.module.state_dict(), os.path.join(args.save_name, '%.2f_model.pth' % epoch))

    return best_model


class SegmentationMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset(device)
        self.count = 0
    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc
    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    # F1-score
    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2*p*r/(p+r)
    # MIOU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        IoU = self.IntersectionOverUnion()
        mIoU = torch.mean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        # mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return IoU
    # FWIOU
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = torch.sum(self.confusionMatrix, dim=1) / (torch.sum(self.confusionMatrix) + 1e-8)
        iu = torch.diag(self.confusionMatrix) / (
                torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) -
                torch.diag(self.confusionMatrix) + 1e-8)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        cm = count.reshape(self.numClass, self.numClass)
        return cm

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self, device):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass), dtype=torch.float64) # int 64 is important, change to float64
        if device=='cuda':
            self.confusionMatrix = self.confusionMatrix.cuda()


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]


def _get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        n_gpu = 0
    elif n_gpu > sys_gpu:
        n_gpu = sys_gpu
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    available_gpus = list(range(n_gpu))
    return device, available_gpus


def init_seg_cd_net():
    device, availble_gpus = _get_available_devices(1)


    # model = IFN.DSIFN(base_model, base_model)
    # model = SiamUnet_conc.SiamUnet_conc(3, 1)
    # model = SiamUnet_diff.SiamUnet_diff(3, 1)
    # model = Unet.Unet(3, 1)
    # model = DTCDSCN.CDNet34(in_channels=3, num_classes=1)

    # model = smp.FFCTLCD(encoder_name="resnet50", encoder_weights="imagenet").to(device)
    model = smp.SegCD(encoder_name="resnet50", encoder_weights="imagenet").to(device)  # STCD
    model = torch.nn.DataParallel(model, device_ids=availble_gpus)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    return model, optimizer


class Dice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred: sigmoid
        # targer: 0, 1
        smooth = 1.
        m1 = pred.view(-1)  # Flatten
        m2 = target.view(-1)  # Flatten
        intersection = (m1 * m2).sum()
        return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# pmask: sigmoid  rmask: 0,1
class BCE_DICE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean')
        # self.ssim = SSIM(window_size=11, size_average=True)
        self.dice = Dice()

    def forward(self, pmask, rmask):
        loss_ce = self.ce(pmask, rmask)
        loss_dice = self.dice(pmask, rmask)
        loss = loss_ce + loss_dice
        return loss


if __name__=="__main__":
    main()