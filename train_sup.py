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


warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--root_path", type=str, default="/home/chrisd/change/STCD/data/", help="root path")
parser.add_argument("--dataset_name", type=str, default="WHU-AB", help="name of the dataset")
parser.add_argument("--CDdataset_name", type=str, default="WHU-AB", help="name of the dataset")
parser.add_argument("--save_name", type=str, default="runs/WHU-Seg-test", help="experiments name")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--load_path", type=str, default="/home/chrisd/change/STCD/resume/", help="load path")
args = parser.parse_args()
print(args)


def main():

    # print(cudnn.is_available())
    torch.backends.cudnn.benchmark = True
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # os.makedirs("saved_models/%s" % args.save_name, exist_ok=True)

    trainloader = DataLoader(
        WHU_Dataset(root_path=args.root_path, dataset=args.dataset_name, train_val='train'),
        batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.n_cpu, drop_last=True)
    val_dataloader = DataLoader(
        WHU_Dataset(root_path=args.root_path, dataset=args.dataset_name,  train_val='val'),
        batch_size=8, shuffle=False, num_workers=1)

    model, optimizer = init_basic_elems()
    # print the model
    load_model = False
    if load_model:
        resume = os.path.join(args.load_path, 'seg_model.pth')
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            model.module.load_state_dict(torch.load(resume))
        else:
            print("=> no checkpoint found at resume")
            print("=> Will start from scratch.")

        generate_label = True
        if generate_label:
            cd_loader = DataLoader(CD_Dataset(root_path=args.root_path, dataset=args.CDdataset_name, train_val='val'),
                                   batch_size=4, shuffle=False, num_workers=4)
            cd_total = SegmentationMetric(numClass=2, device='cuda:0')
            model.eval()
            tbar = tqdm(cd_loader)
            with torch.no_grad():
                for i, (image_A, image_B, label, name) in enumerate(tbar):
                    image_A, image_B, label = image_A.cuda(non_blocking=True), image_B.cuda(non_blocking=True), \
                                              label.cuda(non_blocking=True).unsqueeze(1)
                    pred_A = model(image_A)
                    pred_A = torch.sigmoid(pred_A)
                    pred_A = (pred_A > 0.5).int()  # N C H W

                    pred_B = model(image_B)
                    pred_B = torch.sigmoid(pred_B)
                    pred_B = (pred_B > 0.5).int()  # N C H W

                    # pred_cd = ((torch.abs(pred_B - pred_A)) > 0.2).int()
                    pred_cd = torch.abs(pred_B - pred_A)

                    cd_total.addBatch(pred_cd.cpu(), label.cpu())

                    # pred_cd[pred_cd == 1] = 255
                    # pred_cd = Image.fromarray(pred_cd.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8), mode='L')
                    # pred_cd.save(os.path.join(args.root_path, args.CDdataset_name, 'train', 'pseudo_label', image_name[0]))

            cd_f1 = cd_total.F1score()[1]
            cd_iou = cd_total.IntersectionOverUnion()[1]
            cd_OA = cd_total.OverallAccuracy()
            cd_Pre = cd_total.Precision()[1]
            cd_Rec = cd_total.Recall()[1]
            print('change predictions:  test iou: %.3f, f1 %.3f, Pre %.3f, Rec %.3f, OA %.3f'
                  % (cd_iou, cd_f1, cd_Pre, cd_Rec, cd_OA))

    criterion = BCE_DICE()
    best_model = train_epoch(model, trainloader, val_dataloader, optimizer, criterion, args)
    print("The ending")


def train_epoch(model, trainloader, valloader, optimizer, criterion, args):
    iters = 0
    previous_best = 0.0

    iter_per_epoch = len(trainloader)
    lr_scheduler = Poly(optimizer=optimizer, num_epochs=args.n_epochs, iters_per_epoch=iter_per_epoch)

    writer_dir = os.path.join(args.save_name)
    writer = tensorboard.SummaryWriter(writer_dir)

    for epoch in range(1, args.n_epochs + 1):
        print("\n==> Epoch %i, previous best = %.2f" % (epoch, previous_best))

        model.train()
        wrt_mode = 'train'
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (image, label) in enumerate(tbar):
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()

            pred = model(image)
            pred = torch.sigmoid(pred)
            seg_loss = criterion(pred, label.float())

            seg_loss.backward()
            optimizer.step()
            total_loss += seg_loss.item()
            iters += 1
            lr_scheduler.step(epoch=epoch - 1)

            tbar.set_description('Seg_Loss: %.3f ' % (total_loss / (i + 1)))
            writer.add_scalar(f'{wrt_mode}/Seg_Loss', total_loss / (i + 1), iters)
            del image, label, pred

        model.eval()
        wrt_mode = 'val'
        tbar = tqdm(valloader)

        total_val_loss = 0.0
        with torch.no_grad():
            acc_total = SegmentationMetric(numClass=2, device='cuda:0')
            for i, (image, label) in enumerate(tbar):
                image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True).unsqueeze(1)
                pred = model(image)
                pred = torch.sigmoid(pred)
                val_seg_loss = criterion(pred, label.float())

                total_val_loss += val_seg_loss.item()

                pred = (pred > 0.5)  # N C H W
                acc_total.addBatch(pred.cpu(), label.cpu())

                writer.add_scalar(f'{wrt_mode}/val_Loss', total_val_loss / (i + 1), iters)

        f1 = acc_total.F1score()[1]
        iou = acc_total.IntersectionOverUnion()[1]
        print('epoch %d, train f1 %.3f, iou: %.3f' % (epoch, f1, iou))

        writer.add_scalar(f'{wrt_mode}/seg_f1', f1, epoch)
        writer.add_scalar(f'{wrt_mode}/seg_iou', iou, epoch)

        if iou > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_name, '%.2f_best_model.pth' % previous_best))
            previous_best = iou
            torch.save(model.module.state_dict(), os.path.join(args.save_name, '%.2f_best_model.pth' % iou))
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


def init_basic_elems():
    device, availble_gpus = _get_available_devices(1)

    model = smp.UnetSeg(encoder_name="resnet50", encoder_weights="imagenet").to(device)
    # SETTING THE DEVICE
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