import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageOps, ImageFilter
import glob
import os
import cv2
from torch.utils.data import Dataset
import torchvision
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image
import albumentations as albu
import torch



totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
rcrop = torchvision.transforms.RandomCrop(size=256)
resize = torchvision.transforms.Resize(size=256)


def cutout(img_A, img_B, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img_A = np.array(img_A)
        img_B = np.array(img_B)
        mask = np.array(mask)

        img_h, img_w, img_c = img_A.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img_A[y:y + erase_h, x:x + erase_w] = value
        img_B[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img_A = Image.fromarray(img_A.astype(np.uint8))
        img_B = Image.fromarray(img_B.astype(np.uint8))
        # mask = Image.fromarray(mask.astype(np.uint8))

    return img_A, img_B, mask

def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, 'image', img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, 'label', img_name)


def transforms(img, min_max=(0, 1), train_val='train'):
    # if train_val == "train":
    #     img = np.array(img)
    #     albu_trans = get_training_augmentation()
    #     img = albu_trans(image=img)
    #     img = img['image']
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


class WHU_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.train = train_val
        self.total_path = os.path.join(root_path, dataset, train_val)

        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(self.MEAN, self.STD)
        self.data_path = os.path.join(self.total_path, 'list', train_val + ".txt")
        with open(self.data_path, 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.ids[index]

        image_path = os.path.join(self.total_path, 'A', img_name)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)

        image = Image.fromarray(np.uint8(image))
        if self.train == 'train':
            if random.random() < 0.5:
                image = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image)
            image = T.RandomGrayscale(p=0.2)(image)
            image = blur(image, p=0.5)

        image = self.normalize(self.to_tensor(image))

        L_path = os.path.join(self.total_path, 'A_label', img_name)
        label = np.asarray(Image.open(L_path).convert("RGB"), dtype=np.int32)
        if label.ndim == 3:
            label = label[:, :, 0]
        label[label >= 1] = 1
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        return image, label


class CD_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None, reliable=None):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.train = train_val

        self.cd_total_path = os.path.join(root_path, dataset, train_val)
        if reliable == 'reliable':
            self.change_data_path = os.path.join(self.cd_total_path, 'list', 'reliable_ids' + ".txt")
        elif reliable == 'unreliable':
            self.change_data_path = os.path.join(self.cd_total_path, 'list', 'unreliable_ids' + ".txt")
        else:
            self.change_data_path = os.path.join(self.cd_total_path, 'list', train_val + ".txt")
        with open(self.change_data_path, 'r') as f:
            self.change_ids = f.read().splitlines()

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.change_ids)

    def __getitem__(self, index):
        img_name = self.change_ids[index]

        image_A_path = os.path.join(self.cd_total_path, 'A', img_name)
        image_B_path = os.path.join(self.cd_total_path, 'B', img_name)
        image_A = np.asarray(Image.open(image_A_path).convert("RGB"), dtype=float)
        image_B = np.asarray(Image.open(image_B_path).convert("RGB"), dtype=float)

        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))

        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))

        L_path = get_label_path(self.cd_total_path, img_name)
        label = np.asarray(Image.open(L_path).convert("RGB"), dtype=np.int32)
        if label.ndim == 3:
            label = label[:, :, 0]
        label[label >= 1] = 1
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        return image_A, image_B, label, img_name


class FFC_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.train = train_val

        self.total_path = os.path.join(root_path, dataset, train_val)
        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Seg_Dataset files"""
        img_name = self.files[index].split('/')[-1]

        image_A_path = os.path.join(self.total_path, 'A', img_name)
        image_B_path = os.path.join(self.total_path, 'B', img_name)
        image_A = np.asarray(Image.open(image_A_path).convert("RGB"), dtype=np.float32)
        image_B = np.asarray(Image.open(image_B_path).convert("RGB"), dtype=np.float32)
        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))
        if self.train == 'train':
            if random.random() < 0.5:
                image_A = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_A)
                image_B = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_B)
            image_A = T.RandomGrayscale(p=0.2)(image_A)
            image_A = blur(image_A, p=0.5)
            image_B = T.RandomGrayscale(p=0.2)(image_B)
            image_B = blur(image_B, p=0.5)
        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))
        cd_L_path = os.path.join(self.total_path, 'ff_label', img_name)
        cd_label = np.asarray(Image.open(cd_L_path).convert("RGB"), dtype=np.int32)
        if cd_label.ndim == 3:
            cd_label = cd_label[:, :, 0]
        cd_label[cd_label >= 1] = 1
        cd_label = torch.from_numpy(np.array(cd_label, dtype=np.int32)).long()

        return image_A, image_B, cd_label


class SC_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None, semi=None):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.train = train_val
        self.total_path = os.path.join(root_path, dataset, train_val)
        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))
        self.semi = semi

        self.change_data_path = os.path.join(root_path, dataset, train_val, 'list/changed.txt')
        with open(self.change_data_path, 'r') as f:
            self.change_ids = f.read().splitlines()

        if self.semi:
            self.reliable_data_path = os.path.join(root_path, dataset, train_val, 'list/reliable_ids.txt')
            with open(self.reliable_data_path, 'r') as f:
                self.reliable_ids = f.read().splitlines()

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Seg_Dataset files"""
        img_name = self.files[index].split('/')[-1]
        image_path_A = os.path.join(self.total_path, 'A', img_name)

        s_label_path = os.path.join(self.total_path, 'A_label', img_name)
        # label_show = Image.open(s_label_path).convert("RGB")
        # label_show.show()

        s_label_A = np.asarray(Image.open(s_label_path).convert("RGB"), dtype=np.int32)
        if s_label_A.ndim == 3:
            s_label_A = s_label_A[:, :, 0]
        s_label_A[s_label_A >= 1] = 1
        s_label_A = torch.from_numpy(np.array(s_label_A, dtype=np.int32)).long()

        nc_label_path = os.path.join(self.total_path, 'A_label', '3.tif')
        nc_label = np.asarray(Image.open(nc_label_path).convert("RGB"), dtype=np.int32)
        if nc_label.ndim == 3:
            nc_label = nc_label[:, :, 0]
        nc_label[nc_label >= 1] = 1
        nc_label = torch.from_numpy(np.array(nc_label, dtype=np.int32)).long()

        if img_name in self.change_ids:
            image_path_B = os.path.join(self.total_path, 'WHU-A', img_name.replace('tif', 'png'))
            c_label = s_label_A
            s_label_B = nc_label
        else:
            image_path_B = image_path_A
            c_label = nc_label
            s_label_B = s_label_A

        image_A = np.asarray(Image.open(image_path_A).convert("RGB"), dtype=np.float32)
        image_B = np.asarray(Image.open(image_path_B).convert("RGB"), dtype=np.float32)
        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))
        if self.train == 'train':
            if random.random() < 0.5:
                image_A = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_A)
                image_B = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_B)
            image_A = T.RandomGrayscale(p=0.2)(image_A)
            image_A = blur(image_A, p=0.5)
            image_B = T.RandomGrayscale(p=0.2)(image_B)
            image_B = blur(image_B, p=0.5)

        # image_A.show()
        # image_B.show()
        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))


        if not self.semi:
            return image_A, image_B, s_label_A, s_label_B, c_label
        else:
            if img_name in self.reliable_ids:
                image_CA_path = os.path.join(self.total_path, 'A', img_name)
                image_CB_path = os.path.join(self.total_path, 'B', img_name)
                change_label_path = os.path.join(self.total_path, 'pseudo_label', img_name)
            else:
                image_CA_path = os.path.join(self.total_path, 'B', img_name)
                image_CB_path = os.path.join(self.total_path, 'B', img_name)
                change_label_path = os.path.join(self.total_path, 'label', '0.tif')

            image_CA = np.asarray(Image.open(image_CA_path).convert("RGB"), dtype=np.float32)
            image_CB = np.asarray(Image.open(image_CB_path).convert("RGB"), dtype=np.float32)
            image_CA = Image.fromarray(np.uint8(image_CA))
            image_CB = Image.fromarray(np.uint8(image_CB))

            change_label = np.asarray(Image.open(change_label_path).convert("RGB"), dtype=np.int32)

            if self.train == 'train':
                if random.random() < 0.8:
                    image_CA = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_CA)
                    image_CB = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_CB)
                image_CA = T.RandomGrayscale(p=0.2)(image_CA)
                image_CB = T.RandomGrayscale(p=0.2)(image_CB)
                image_CA = blur(image_CA, p=0.5)
                image_CB = blur(image_CB, p=0.5)
                # image_CA, image_CB, change_label = cutout(image_CA, image_CB, change_label, p=0.5)

            # label_show = Image.open(change_label_path).convert("RGB")
            # label_show.show()
            # image_CA.show()
            # image_CB.show()

            image_CA = self.normalize(self.to_tensor(image_CA))
            image_CB = self.normalize(self.to_tensor(image_CB))

            if change_label.ndim == 3:
                change_label = change_label[:, :, 0]
            change_label[change_label >= 1] = 1
            change_label = torch.from_numpy(np.array(change_label, dtype=np.int32)).long()

            return image_A, image_B, s_label_A, s_label_B, c_label, image_CA, image_CB, change_label, img_name


class PSE_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.train = train_val

        self.total_path = os.path.join(root_path, dataset, train_val)
        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """Seg_Dataset files"""
        img_name = self.files[index].split('/')[-1]

        image_A_path = os.path.join(self.total_path, 'A', img_name)
        image_B_path = os.path.join(self.total_path, 'B', img_name)
        image_A = np.asarray(Image.open(image_A_path).convert("RGB"), dtype=np.float32)
        image_B = np.asarray(Image.open(image_B_path).convert("RGB"), dtype=np.float32)
        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))
        if self.train == 'train':
            if random.random() < 0.5:
                image_A = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_A)
                image_B = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_B)
            image_A = T.RandomGrayscale(p=0.2)(image_A)
            image_A = blur(image_A, p=0.5)
            image_B = T.RandomGrayscale(p=0.2)(image_B)
            image_B = blur(image_B, p=0.5)
        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))
        cd_L_path = os.path.join(self.total_path, 'pseudo_label_WHU', img_name)
        cd_label = np.asarray(Image.open(cd_L_path).convert("RGB"), dtype=np.int32)
        if cd_label.ndim == 3:
            cd_label = cd_label[:, :, 0]
        cd_label[cd_label >= 1] = 1
        cd_label = torch.from_numpy(np.array(cd_label, dtype=np.int32)).long()

        return image_A, image_B, cd_label


class LEVIR_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.train = train_val
        self.total_path = os.path.join(root_path, 'WHU-AB', train_val)
        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))

        self.pse_change_data_path = os.path.join(root_path, 'WHU-AB', train_val, 'list/changed.txt')
        with open(self.pse_change_data_path, 'r') as f:
            self.change_ids = f.read().splitlines()

        # self.reliable_data_path = os.path.join(root_path, dataset, train_val, 'list/reliable_ids.txt')
        # with open(self.reliable_data_path, 'r') as f:
        #     self.reliable_ids = f.read().splitlines()

        self.total_change_path = os.path.join(root_path, dataset, train_val)
        self.change_files = sorted(glob.glob(self.total_change_path + "/A/*.*"))

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.change_files)

    def __getitem__(self, index):
        """Seg_Dataset files"""
        if index > 5939:
            seg_index = index % 5939
        else:
            seg_index = index
        img_name = self.files[seg_index].split('/')[-1]
        image_path_A = os.path.join(self.total_path, 'A', img_name)

        s_label_path = os.path.join(self.total_path, 'A_label', img_name)
        # label_show = Image.open(s_label_path).convert("RGB")
        # label_show.show()

        s_label_A = np.asarray(Image.open(s_label_path).convert("RGB"), dtype=np.int32)
        if s_label_A.ndim == 3:
            s_label_A = s_label_A[:, :, 0]
        s_label_A[s_label_A >= 1] = 1
        s_label_A = torch.from_numpy(np.array(s_label_A, dtype=np.int32)).long()

        nc_label_path = os.path.join(self.total_path, 'A_label', '3.tif')
        nc_label = np.asarray(Image.open(nc_label_path).convert("RGB"), dtype=np.int32)
        if nc_label.ndim == 3:
            nc_label = nc_label[:, :, 0]
        nc_label[nc_label >= 1] = 1
        nc_label = torch.from_numpy(np.array(nc_label, dtype=np.int32)).long()

        if img_name in self.change_ids:
            image_path_B = os.path.join(self.total_path, 'WHU-A', img_name.replace('tif', 'png'))
            c_label = s_label_A
            s_label_B = nc_label
        else:
            image_path_B = image_path_A
            c_label = nc_label
            s_label_B = s_label_A

        image_A = np.asarray(Image.open(image_path_A).convert("RGB"), dtype=np.float32)
        image_B = np.asarray(Image.open(image_path_B).convert("RGB"), dtype=np.float32)
        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))
        if self.train == 'train':
            if random.random() < 0.5:
                image_A = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_A)
                image_B = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_B)
            image_A = T.RandomGrayscale(p=0.2)(image_A)
            image_A = blur(image_A, p=0.5)
            image_B = T.RandomGrayscale(p=0.2)(image_B)
            image_B = blur(image_B, p=0.5)

        # image_A.show()
        # image_B.show()
        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))

        change_img_name = self.change_files[index].split('/')[-1]

        image_CA_path = os.path.join(self.total_change_path, 'A', change_img_name)
        image_CB_path = os.path.join(self.total_change_path, 'B', change_img_name)
        change_label_path = os.path.join(self.total_change_path, 'pseudo_label_WHU', change_img_name)

        # if change_img_name in self.reliable_ids:
        #     image_CA_path = os.path.join(self.total_change_path, 'A', change_img_name)
        #     image_CB_path = os.path.join(self.total_change_path, 'B', change_img_name)
        #     change_label_path = os.path.join(self.total_change_path, 'pseudo_label', change_img_name)
        # else:
        #     image_CA_path = os.path.join(self.total_change_path, 'B', change_img_name)
        #     image_CB_path = os.path.join(self.total_change_path, 'B', change_img_name)
        #     change_label_path = os.path.join(self.total_change_path, 'label', 'train_1_6.png')

        image_CA = np.asarray(Image.open(image_CA_path).convert("RGB"), dtype=np.float32)
        image_CB = np.asarray(Image.open(image_CB_path).convert("RGB"), dtype=np.float32)
        image_CA = Image.fromarray(np.uint8(image_CA))
        image_CB = Image.fromarray(np.uint8(image_CB))

        change_label = np.asarray(Image.open(change_label_path).convert("RGB"), dtype=np.int32)

        if self.train == 'train':
            if random.random() < 0.8:
                image_CA = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_CA)
                image_CB = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_CB)
            image_CA = T.RandomGrayscale(p=0.2)(image_CA)
            image_CB = T.RandomGrayscale(p=0.2)(image_CB)
            image_CA = blur(image_CA, p=0.5)
            image_CB = blur(image_CB, p=0.5)
            # image_CA, image_CB, change_label = cutout(image_CA, image_CB, change_label, p=0.5)

        # label_show = Image.open(change_label_path).convert("RGB")
        # label_show.show()
        # image_CA.show()
        # image_CB.show()

        image_CA = self.normalize(self.to_tensor(image_CA))
        image_CB = self.normalize(self.to_tensor(image_CB))

        if change_label.ndim == 3:
            change_label = change_label[:, :, 0]
        change_label[change_label >= 1] = 1
        change_label = torch.from_numpy(np.array(change_label, dtype=np.int32)).long()

        return image_A, image_B, s_label_A, s_label_B, c_label, image_CA, image_CB, change_label, change_img_name

