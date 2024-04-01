import torch.utils.data as data
from torch.utils.data import DataLoader
import os, sys
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import yaml
from tqdm import trange, tqdm
import glob
import json
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix
from pytorch3d.io import load_objs_as_meshes, IO
import trimesh
import objaverse
from utils import *

np.set_printoptions(threshold=np.inf)

def get_coco_image_fn_list(data_path, COCO_IMAGE_ROOT):
    if os.path.exists(data_path):
        print("Coco exists, loading data ========>>>>>>>>>")
        return read_pickle(data_path)
    else:
        print("Coco doesn't exist, processing data ========>>>>>>>>>")
        img_list = os.listdir(COCO_IMAGE_ROOT)
        img_list = [os.path.join(COCO_IMAGE_ROOT, img) for img in img_list if img.endswith('.jpg')]
        save_pickle(img_list, data_path)
        return img_list

def get_background_image_coco(img_path, h, w):
    # bg_img = cv2.imread(img_path)
    bg_img = Image.open(img_path)
    bg_img = np.asarray(bg_img, dtype=np.uint8)

    h1, w1 = bg_img.shape[:2]
    if h1 > h and w1 > w:
        hb = np.random.randint(0, h1 - h)
        wb = np.random.randint(0, w1 - w)
        bg_img = bg_img[hb:hb + h, wb:wb + w]
    else:
        bg_img = cv2.resize(bg_img, (w,h), interpolation=cv2.INTER_LINEAR)
    if len(bg_img.shape)==2:
        bg_img = np.repeat(bg_img[:,:,None],3,2)
    return bg_img[:,:,:3]

class Dataset_Loader_Objaverse_stereo(data.Dataset):
    def __init__(self, cfg, mode, train_indices=None):
        self.cfg = cfg
        self.mode = mode
        self.data_path = self.cfg["OBJAVERSE"]["DATA_PATH"]
        self.img_path = os.path.join(self.data_path, "views_release")

        print("Loading data of images =======>>>>>>>>")
        self.folder_list = os.listdir(self.img_path)
        if self.mode == "train":
            if train_indices is None:
                self.folder_list = self.folder_list[:-self.cfg["DATA"]["UNSEEN_NUM"]]
            else:
                self.folder_list = [self.folder_list[idx] for idx in train_indices]

        print("Lenth of %s dataset:" % (self.mode), len(self.folder_list))

        print("Loading background images")
        self.bg_img_list = get_coco_image_fn_list(self.cfg["OBJAVERSE"]["COCO_PATH_FILE"], self.cfg["OBJAVERSE"]["COCO_IMAGE_ROOT"])
        print("Lenth of background images:", len(self.bg_img_list))

        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self.cfg['DATA']['PIXEL_MEAN'],
                    self.cfg['DATA']['PIXEL_STD']),
            ]
        )

    def __len__(self):
        return len(self.folder_list)

    def inplane_augmentation(self, img, mask, R, T):
        theta = (2 * torch.rand(1) - 1) * 180
        center = (img.shape[1]/2, img.shape[0]/2)

        aug_R = torch.tensor(
                ((torch.cos(theta * np.pi / 180), -torch.sin(theta * np.pi / 180), 0),
                 (torch.sin(theta * np.pi / 180), torch.cos(theta * np.pi / 180), 0),
                 (0, 0, 1)))

        R = aug_R @ R
        T = aug_R @ T

        aug_R_2d = cv2.getRotationMatrix2D(center, -theta.item(), scale=1)
        img = cv2.warpAffine(img, aug_R_2d, (img.shape[1], img.shape[0]))
        mask = cv2.warpAffine(mask, aug_R_2d, (img.shape[1], img.shape[0]))

        return img, mask, R, T

    def load_pose_info(self, img_path):
        pose_path = img_path.replace(".png", ".npy")
        pose = np.load(pose_path)

        R = torch.from_numpy(pose[:3, :3]).float()
        T = torch.from_numpy(pose[:3, 3:]).float()

        ### from blender to opencv
        R_bcam2cv = torch.tensor(
                ((1, 0,  0),
                 (0, -1, 0),
                 (0, 0, -1))).float()
        R = R_bcam2cv @ R
        T = R_bcam2cv @ T

        return R, T

    def load_info(self, img_paths, index, R_anchor=None):
        img_path = img_paths[index]
        R, T = self.load_pose_info(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = (img[:, :, 3] > 0).astype(np.uint8)
        img = img[:, :, :3].astype(np.uint8)

        ### augmentation of in-plane rotation
        if torch.rand(1) < self.cfg["TRAIN"]["ROTA_RATIO"]:
            img, mask, R_aug, T_aug = self.inplane_augmentation(img, mask, R, T)
        else:
            R_aug, T_aug = R, T

        crop_params = torch.zeros(4).float()

        if mask.sum() > self.cfg["DATA"]["SIZE_THR"] ** 2:
            bbx = np.where(mask>0)
            x_min = int(np.min(bbx[1]))
            y_min = int(np.min(bbx[0]))
            x_max = int(np.max(bbx[1]))
            y_max = int(np.max(bbx[0]))
            bbx = np.asarray([x_min, y_min, x_max, y_max])

            bbx = bbx_resize(bbx, mask.shape[1], mask.shape[0])
            img = crop(img, bbx)
            mask = crop(mask, bbx)

            ### random background
            if torch.rand(1) < self.cfg["TRAIN"]["BG_RATIO"]:
                index = torch.randperm(len(self.bg_img_list))[0]
                bg_img = get_background_image_coco(self.bg_img_list[index], img.shape[0], img.shape[1])

                mask_blur = cv2.GaussianBlur(255 * mask.astype(np.uint8), (3, 3), cv2.BORDER_DEFAULT)[..., None] / 255
                img = img * mask_blur + bg_img * (1 - mask_blur)
                img = img.astype(np.uint8)

        img = self.trans(img)
        mask = torch.from_numpy(mask).float()[None]

        if mask.sum() > self.cfg["DATA"]["SIZE_THR"] ** 2:
            ratio = torch.tensor([float(img.shape[2]) / float(self.cfg["DATA"]["OBJ_SIZE"]), float(img.shape[1]) / float(self.cfg["DATA"]["OBJ_SIZE"])]).float()
            crop_params[:2] = torch.tensor([bbx[0], bbx[1]]).float()
            crop_params[2:] = ratio

        img = resize_pad(img, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.BILINEAR)
        mask = resize_pad(mask, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.NEAREST)

        return img, mask, R_aug, T_aug, R, T, crop_params


    def __getitem__(self, idx):
        folder = self.folder_list[idx]
        img_path = glob.glob(os.path.join(self.img_path, folder, "*.png"))
        while len(img_path) == 0:
            idx = torch.randperm(len(self.folder_list))[0]
            folder = self.folder_list[idx]
            img_path = glob.glob(os.path.join(self.img_path, folder, "*.png"))

        indices = torch.randperm(len(img_path))[:2]

        src_img, src_mask, src_R, src_T, src_R_init, src_T_init, src_crop_params = self.load_info(img_path, indices[0])
        ref_img, ref_mask, ref_R, ref_T, ref_R_init, ref_T_init, ref_crop_params = self.load_info(img_path, indices[1])

        dis_init = torch.arccos((torch.sum(src_R_init.view(-1) * ref_R_init.view(-1)).clamp(-1, 3) - 1) / 2) * 180. / np.pi

        data = {
            "src_img":src_img, "src_mask":src_mask, "src_R":src_R, "src_T":src_T, "src_crop_params":src_crop_params,
            "ref_img":ref_img, "ref_mask":ref_mask, "ref_R":ref_R, "ref_T":ref_T, "ref_crop_params":ref_crop_params,
            "dis_init":dis_init
        }
        return data

class Dataset_Loader_Objaverse_stereo_test(data.Dataset):
    def __init__(self, cfg, test_indices=None, transform=None):
        self.cfg = cfg
        self.data_path = self.cfg["OBJAVERSE"]["DATA_PATH"]
        self.img_path = os.path.join(self.data_path, "views_release")

        print("Loading data of images =======>>>>>>>>")

        self.folder_list = os.listdir(self.img_path)

        if test_indices is None:
            self.folder_list = self.folder_list[-self.cfg["DATA"]["UNSEEN_NUM"]:]
        else:
            self.folder_list = [self.folder_list[idx] for idx in test_indices]

        self.test_data_preprocess()

        print("Lenth of dataset:", len(self.folder_list))

        print("Loading background images")
        self.bg_img_list = get_coco_image_fn_list(self.cfg["OBJAVERSE"]["COCO_PATH_FILE"], self.cfg["OBJAVERSE"]["COCO_IMAGE_ROOT"])
        print("Lenth of background images:", len(self.bg_img_list))

        if transform is None:
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.cfg['DATA']['PIXEL_MEAN'],
                        self.cfg['DATA']['PIXEL_STD']),
                ]
            )
        else:
            self.trans = transform

        self.meta_infos, self.thetas = [], []
        print("Loading testing data")
        for folder in self.folder_list:
            if not os.path.exists(os.path.join(self.img_path, folder, "pairs.txt")):
                raise RuntimeError("Pairing file does not exist")

            with open(os.path.join(self.img_path, folder, "pairs.txt"), 'r') as f:
                for path in f:
                    meta_info = {}
                    img_path = path.rstrip()
                    meta_info["folder"] = folder
                    meta_info["src"] = img_path.split(" ")[0]
                    meta_info["ref"] = img_path.split(" ")[1]
                    self.meta_infos.append(meta_info)
            f.close()

            if not os.path.exists(os.path.join(self.img_path, folder, "inplane_theta.txt")):
                raise RuntimeError("Theta file does not exist")

            self.thetas.append(np.loadtxt(os.path.join(self.img_path, folder, "inplane_theta.txt")))

        self.thetas = np.concatenate(self.thetas)

    def __len__(self):
        return len(self.meta_infos)

    def load_test_scenes(self, path):
        if not os.path.exists(path):
            raise RuntimeError("Pairing file does not exist")
        folder_list = []
        with open(path, 'r') as f:
            for path in f:
                folder_list.append(path[:-1])
        f.close()
        return folder_list

    def test_data_preprocess(self):
        for folder in self.folder_list:
            if not os.path.exists(os.path.join(self.img_path, folder, "pairs.txt")):
                print("Processing " + folder)
                f = open(os.path.join(self.img_path, folder, "pairs.txt"),'w')

                img_paths = glob.glob(os.path.join(self.img_path, folder, "*.png"))
                pose = [self.load_pose_info(img_path) for img_path in img_paths]
                Rs, Ts = zip(*pose)
                Rs, Ts = torch.stack(Rs), torch.stack(Ts)
                geo_dis = torch.arccos((torch.sum(Rs.view(-1, 1, 9) * Rs.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2) * 180. / np.pi
                for i in range(Rs.shape[0]):
                    indices = torch.nonzero(geo_dis[i] < self.cfg["DATA"]["VIEW_THR"]).squeeze(-1)
                    src_img = img_paths[i].split("/")[-1]
                    ref_imgs = [img_paths[index].split("/")[-1] for index in indices]

                    for ref_img in ref_imgs:
                    	f.write(src_img + " " + ref_img + "\n")
                f.close()

            if not os.path.exists(os.path.join(self.img_path, folder, "inplane_theta.txt")):
                thetas = []
                with open(os.path.join(self.img_path, folder, "pairs.txt"), 'r') as f:
                    for path in f:
                        thetas.append(torch.rand(1).item())
                f.close()
                thetas = np.asarray(thetas)
                np.savetxt(os.path.join(self.img_path, folder, "inplane_theta.txt"), thetas)


    def inplane_augmentation(self, theta, img, mask, R, T):
        angle = (2 * theta - 1) * 180
        center = (img.shape[1]/2, img.shape[0]/2)

        aug_R = torch.tensor(
                ((np.cos(angle * np.pi / 180), -np.sin(angle * np.pi / 180), 0),
                 (np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180), 0),
                 (0, 0, 1))).float()

        R = aug_R @ R
        T = aug_R @ T

        aug_R_2d = cv2.getRotationMatrix2D(center, -angle, scale=1)
        img = cv2.warpAffine(img, aug_R_2d, (img.shape[1], img.shape[0]))
        mask = cv2.warpAffine(mask, aug_R_2d, (img.shape[1], img.shape[0]))

        return img, mask, R, T

    def load_pose_info(self, img_path):
        pose_path = img_path.replace(".png", ".npy")
        pose = np.load(pose_path)

        R = torch.from_numpy(pose[:3, :3]).float()
        T = torch.from_numpy(pose[:3, 3:]).float()

        ### from blender to opencv
        R_bcam2cv = torch.tensor(
                ((1, 0,  0),
                 (0, -1, 0),
                 (0, 0, -1))).float()
        R = R_bcam2cv @ R
        T = R_bcam2cv @ T

        return R, T

    def load_info(self, img_path, theta=None):
        R, T = self.load_pose_info(img_path)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = (img[:, :, 3] > 0).astype(np.uint8)
        img = img[:, :, :3]

        crop_params = torch.zeros(4).float()

        if mask.sum() > self.cfg["DATA"]["SIZE_THR"] ** 2:
            if theta is not None:
                img, mask, R, T = self.inplane_augmentation(theta, img, mask, R, T)

            if self.cfg["DATA"]["OBJ_SIZE"] is not None:
                bbx = np.where(mask>0)
                x_min = int(np.min(bbx[1]))
                y_min = int(np.min(bbx[0]))
                x_max = int(np.max(bbx[1]))
                y_max = int(np.max(bbx[0]))
                bbx = np.asarray([x_min, y_min, x_max, y_max])

                bbx = bbx_resize(bbx, mask.shape[1], mask.shape[0])
                img = crop(img, bbx)
                mask = crop(mask, bbx)

            ### random background
            index = torch.randperm(len(self.bg_img_list))[0]
            bg_img = get_background_image_coco(self.bg_img_list[index], img.shape[0], img.shape[1])

            mask_blur = cv2.GaussianBlur(255 * mask.astype(np.uint8), (3, 3), cv2.BORDER_DEFAULT)[..., None] / 255
            img = img * mask_blur + bg_img * (1 - mask_blur)
            img = img.astype(np.uint8)

        img = self.trans(img)
        mask = torch.from_numpy(mask).float()[None]

        if self.cfg["DATA"]["OBJ_SIZE"] is not None and mask.sum() > self.cfg["DATA"]["SIZE_THR"] ** 2:
            ratio = torch.tensor([float(img.shape[2]) / float(self.cfg["DATA"]["OBJ_SIZE"]), float(img.shape[1]) / float(self.cfg["DATA"]["OBJ_SIZE"])]).float()
            crop_params[:2] = torch.tensor([bbx[0], bbx[1]]).float()
            crop_params[2:] = ratio

        if self.cfg["DATA"]["OBJ_SIZE"] is not None:
            img = resize_pad(img, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.BILINEAR)
            mask = resize_pad(mask, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.NEAREST)

        return img, mask, R, T, crop_params

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        theta = self.thetas[idx]
        src_img_path = os.path.join(self.img_path, meta_info["folder"], meta_info["src"])
        ref_img_path = os.path.join(self.img_path, meta_info["folder"], meta_info["ref"])

        src_img, src_mask, src_R, src_T, src_crop_params = self.load_info(src_img_path, theta=None)
        ref_img, ref_mask, ref_R, ref_T, ref_crop_params = self.load_info(ref_img_path, theta=theta)

        data = {
            "src_img":src_img, "src_mask":src_mask, "src_R":src_R, "src_T":src_T, "src_crop_params":src_crop_params,
            "ref_img":ref_img, "ref_mask":ref_mask, "ref_R":ref_R, "ref_T":ref_T, "ref_crop_params":ref_crop_params,
        }

        return data

class Dataset_Loader_LINEMOD_stereo(data.Dataset):
    def __init__(self, cfg, clsID, transform=None):
        self.cfg = cfg

        self.jitter_scale = (1.0 / (1.0 + cfg["DATA"]["NOISE_LEVEL"]), 1.0 * (1.0 + cfg["DATA"]["NOISE_LEVEL"]))
        self.jitter_trans = (-0.5 * cfg["DATA"]["NOISE_LEVEL"], 0.5 * cfg["DATA"]["NOISE_LEVEL"])

        self.src_path = os.path.join(self.cfg['LINEMOD']['META_DIR'], 'src_images_test_pkl', '%06d.pkl' % (clsID))

        if transform is None:
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.cfg['DATA']['PIXEL_MEAN'],
                        self.cfg['DATA']['PIXEL_STD']),
                ]
            )
        else:
            self.trans = transform

        print(">>>>>>> Loading source data")
        self.meta_info = {}
        src_imgs, src_masks, src_Ks, src_Rs, src_Ts, src_bbxs, src_ids = [], [], [], [], [], [], []
        ref_info = {}

        with open(self.src_path, 'rb') as f:
            self.meta_info = pickle.load(f)
        f.close()

        self.test_data_preprocess(self.src_path, self.meta_info)

        self.pair_indices = np.loadtxt(self.src_path.replace(".pkl", "_pairs.txt"))

        with open(cfg["LINEMOD"]["BBOX_FILE"], 'r') as f:
            self.bbox_3d = json.load(f)
        f.close()

    def __len__(self):
        return self.pair_indices.shape[0]

    def test_data_preprocess(self, src_path, src_info):
        if not os.path.exists(src_path.replace(".pkl", "_pairs.txt")):
            print("Processing " + src_path)

            Rs = torch.from_numpy(np.asarray(src_info["Rs"])).float()
            eulers = matrix_to_euler_angles(Rs, "ZXZ") ### in-plane ele azi
            eulers[:, 0] = 0
            Rs = euler_angles_to_matrix(eulers, "ZXZ")

            geo_dis = torch.arccos((torch.sum(Rs.view(-1, 1, 9) * Rs.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2) * 180. / np.pi
            indices = torch.nonzero(geo_dis < self.cfg["DATA"]["VIEW_THR"])#.squeeze(-1)
            indices = indices[torch.randperm(indices.shape[0])[:1000]]

            np.savetxt(src_path.replace(".pkl", "_pairs.txt"), indices.numpy())

    def load_info(self, idx):
        img = self.meta_info["imgs"][idx].astype(np.uint8)
        mask = self.meta_info["masks"][idx].astype(np.uint8)
        bbx = self.meta_info["bbxs"][idx]
        R = torch.from_numpy(self.meta_info["Rs"][idx]).float()
        T = torch.from_numpy(self.meta_info["Ts"][idx]).float()

        bbx = bbx_resize(bbx, mask.shape[1], mask.shape[0], scale_ratio=1.0) #1.2

        if self.cfg["DATA"]["JITTER"] is True:
            bbx = jitter_bbox(bbx, self.jitter_scale, self.jitter_trans, img.shape[:2])

        img = crop(img, bbx)
        mask = crop(mask, bbx)
        img = self.trans(img)
        mask = torch.from_numpy(mask).float()[None]

        if self.cfg["DATA"]["OBJ_SIZE"] is not None:
            ratio = torch.tensor([float(img.shape[2]) / float(self.cfg["DATA"]["OBJ_SIZE"]), float(img.shape[1]) / float(self.cfg["DATA"]["OBJ_SIZE"])]).float()
            img = resize_pad(img, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.BILINEAR)
            mask = resize_pad(mask, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.NEAREST)
        else:
            ratio = torch.ones(2).float()

        crop_params = torch.tensor([bbx[0], bbx[1]]).float()
        crop_params = torch.cat([crop_params, ratio])

        return img, mask, R, T, crop_params

    def __getitem__(self, idx):
        src_id = int(self.pair_indices[idx, 0])
        ref_id = int(self.pair_indices[idx, 1])

        src_img, src_mask, src_R, src_T, src_crop_params = self.load_info(src_id)
        ref_img, ref_mask, ref_R, ref_T, ref_crop_params = self.load_info(ref_id)

        data = {
            "src_img":src_img, "src_mask":src_mask, "src_R":src_R, "src_T":src_T, "src_crop_params":src_crop_params,
            "ref_img":ref_img, "ref_mask":ref_mask, "ref_R":ref_R, "ref_T":ref_T, "ref_crop_params":ref_crop_params,
        }

        return data

class Dataset_Loader_LINEMOD_stereo_train(data.Dataset):
    def __init__(self, cfg, clsIDs, transform=None):
        self.cfg = cfg

        self.src_path = [os.path.join(self.cfg['LINEMOD']['META_DIR'], 'src_images_test_pkl', '%06d.pkl' % (clsID)) for clsID in clsIDs]

        if transform is None:
            self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.cfg['DATA']['PIXEL_MEAN'],
                        self.cfg['DATA']['PIXEL_STD']),
                ]
            )
        else:
            self.trans = transform

        self.meta_info = {}
        self.pair_indices, self.ids = [], []
        for idx, clsID in enumerate(clsIDs):
            print(">>>>>>> Loading source data of %06d" % (clsID))

            with open(self.src_path[idx], 'rb') as f:
                self.meta_info["%06d" % (clsID)] = pickle.load(f)
            f.close()
            self.data_preprocess(self.src_path[idx], self.meta_info["%06d" % (clsID)], rebuild=False)

            pair_indices = np.loadtxt(self.src_path[idx].replace(".pkl", "_pairs.txt"))

            self.ids.append(clsID*np.ones([pair_indices.shape[0]]))
            self.pair_indices.append(pair_indices)

        self.ids = np.concatenate(self.ids, axis=0)
        self.pair_indices = np.concatenate(self.pair_indices, axis=0)

        print("Loading is done, the lenth of training data is:", self.pair_indices.shape[0])

    def __len__(self):
        return self.pair_indices.shape[0]

    def data_preprocess(self, src_path, src_info, rebuild=False):
        if not os.path.exists(src_path.replace(".pkl", "_pairs.txt")) or rebuild is True:
            print("Processing " + src_path)

            Rs = torch.from_numpy(np.asarray(src_info["Rs"])).float()
            eulers = matrix_to_euler_angles(Rs, "ZXZ") ### in-plane ele azi
            eulers[:, 0] = 0
            Rs = euler_angles_to_matrix(eulers, "ZXZ")

            geo_dis = torch.arccos((torch.sum(Rs.view(-1, 1, 9) * Rs.view(1, -1, 9), dim=-1).clamp(-1, 3) - 1) / 2) * 180. / np.pi
            indices = torch.nonzero(geo_dis < self.cfg["DATA"]["VIEW_THR"])#.squeeze(-1)
            indices = indices[torch.randperm(indices.shape[0])[:20000]]

            np.savetxt(src_path.replace(".pkl", "_pairs.txt"), indices.numpy())

    def load_info(self, idx, meta_info):
        img = meta_info["imgs"][idx].astype(np.uint8)
        mask = meta_info["masks"][idx].astype(np.uint8)
        bbx = meta_info["bbxs"][idx]
        R = torch.from_numpy(meta_info["Rs"][idx]).float()
        T = torch.from_numpy(meta_info["Ts"][idx]).float()

        h, w = img.shape[:2]

        bbx = bbx_resize(bbx, mask.shape[1], mask.shape[0])
        img = crop(img, bbx)
        mask = crop(mask, bbx)

        crop_center = (bbx[:2] + bbx[2:]) / 2
        cc = (2 * crop_center / min(h, w)) - 1
        crop_width = 2 * (bbx[2] - bbx[0]) / min(h, w)

        crop_params = torch.tensor([-cc[0], -cc[1], crop_width]).float()

        img = self.trans(img)
        mask = torch.from_numpy(mask).float()[None]

        if self.cfg["DATA"]["OBJ_SIZE"] is not None:
            img = resize_pad(img, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.BILINEAR)
            mask = resize_pad(mask, self.cfg["DATA"]["OBJ_SIZE"], transforms.InterpolationMode.NEAREST)

        return img, mask, R, T, crop_params

    def __getitem__(self, idx):
        clsID = self.ids[idx]
        meta_info = self.meta_info["%06d" % (clsID)]

        src_id = int(self.pair_indices[idx, 0])
        ref_id = int(self.pair_indices[idx, 1])

        src_img, src_mask, src_R, src_T, src_crop_params = self.load_info(src_id, meta_info)
        ref_img, ref_mask, ref_R, ref_T, ref_crop_params = self.load_info(ref_id, meta_info)

        data = {
            "src_img":src_img, "src_mask":src_mask, "src_R":src_R, "src_T":src_T, "src_crop_params":src_crop_params,
            "ref_img":ref_img, "ref_mask":ref_mask, "ref_R":ref_R, "ref_T":ref_T, "ref_crop_params":ref_crop_params,
        }

        return data
