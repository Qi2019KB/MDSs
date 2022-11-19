# -*- coding: utf-8 -*-
import os
import random
import copy
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.udaap.utils_augment import Augment


class MultiDataset(torch.utils.data.Dataset):
    def update(self, pseudoArray):
        self.dataArray = copy.deepcopy(self.dataArray_reset)
        for pseudoItem in pseudoArray:
            if pseudoItem["enable"] > 0:
                kIdx = pseudoItem["kpID"].split("_")[-1]
                kIdx_len = len(kIdx)
                imageID = pseudoItem["kpID"][0:(-1-kIdx_len)]
                # imageID, kIdx = pseudoItem["kpID"].split("_")
                dataItem = [item for item in self.dataArray if item["imageID"] == imageID][0]
                dataItem["kps"][int(kIdx)] = [pseudoItem["coord"][0], pseudoItem["coord"][1], pseudoItem["enable"]]

    def __init__(self, dsType, dataArray, means, stds, multiCount=2, isAug=False, isDraw=False, **kwargs):
        self.dsType, self.dataArray, self.means, self.stds, self.isAug, self.isDraw = dsType, copy.deepcopy(dataArray), means, stds, isAug, isDraw
        self.multiCount = multiCount
        self.dataArray_reset = copy.deepcopy(dataArray)
        self.inpRes, self.outRes, self.basePath, self.imgType = kwargs['inpRes'], kwargs['outRes'], kwargs['basePath'], kwargs['imgType']
        self.useFlip, self.useOcc, self.sf, self.rf = kwargs['useFlip'], kwargs['useOcclusion'], kwargs['scaleRange'], kwargs['rotRange']
        self.useOcc_ema, self.sf_ema, self.rf_ema = kwargs['useOcclusion_ema'], kwargs['scaleRange_ema'], kwargs['rotRange_ema']
        if self.isAug:
            if self.useOcc:
                self.augmentor = Augment(num_occluder=kwargs['numOccluder'])  # student专用。从voc2012数据集中拿到一些mask形状，来遮挡图像。
            if self.useOcc_ema:
                self.augmentor_ema = Augment(num_occluder=kwargs['numOccluder_ema'])  # teacher专用。从voc2012数据集中拿到一些mask形状，来遮挡图像。

    def __getitem__(self, idx):
        # region 1.数据预处理（含图像resize）
        annObj = self.dataArray[idx]
        img, kps, kps_ori_test, dataID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["id"]  # H*W*C
        img, kps, _ = proc.image_resize(img, kps, self.inpRes)  # H*W*C
        _, kps_test, _ = proc.image_resize(img, kps_ori_test, self.inpRes)  # 测试用。用于验证选用的伪标签的质量。
        kpsMap_test = torch.from_numpy(np.array(kps_test).astype(np.float32))  # 测试用。用于验证选用的伪标签的质量。
        # endregion
        # region 2. 生成原样本对应数据
        imgMap_ori = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        kpsMap_ori = torch.from_numpy(np.array(kps).astype(np.float32))
        kpsHeatmap_ori, kpsMap = proc.kps_heatmap(kpsMap_ori, imgMap_ori.shape, self.inpRes, self.outRes)
        kpsWeight_ori = kpsMap[:, 2].clone()
        angle_ori, scale_ori = torch.tensor(0.), torch.tensor(self.inpRes / 200.0)
        cenMap_ori = torch.tensor(proc.center_calculate(img))
        # endregion
        # region 3. 生成增广样本对应数据
        # region 3.1 增广样本数据准备
        imgMap = imgMap_ori.clone()  # H*W*C ==> C*H*W
        kpsMap = kpsMap_ori.clone()
        angle, scale, cenMap = angle_ori.clone(), scale_ori.clone(), cenMap_ori.clone()
        angle_array, scale_array, imgMap_array, kpsMap_array, cenMap_array, isflip_array = [angle], [scale], [imgMap], [kpsMap], [cenMap], [False]
        for mulIdx in range(self.multiCount-1):
            angle_array.append(angle.clone())
            scale_array.append(scale.clone())
            imgMap_array.append(imgMap.clone())
            kpsMap_array.append(kpsMap.clone())
            cenMap_array.append(cenMap.clone())
            isflip_array.append(False)
        # endregion
        # region 3.2 增广样本数据处理
        kpsHeatmap_array, kpsWeight_array, warpmat_array = [], [], []
        for idx in range(self.multiCount):
            # region 3.2.1 数据准备
            imgMap, kpsMap, cenMap = imgMap_array[idx], kpsMap_array[idx], cenMap_array[idx]
            scale, angle, isflip = scale_array[idx], angle_array[idx], isflip_array[idx]
            # endregion
            # region 3.2.2 数据增强
            if self.isAug:
                # 随机水平翻转
                imgMap, kpsMap, cenMap, isflip = aug.fliplr(imgMap, kpsMap, cenMap, prob=0.5)
                # 随机加噪（随机比例去均值）
                imgMap = aug.noisy_mean(imgMap)
                # 随机仿射变换（随机缩放、随机旋转）
                imgMap, kpsMap, scale, angle = aug.affine(imgMap, kpsMap, cenMap, scale, self.sf, angle, self.rf, [self.inpRes, self.inpRes])
                # 随机遮挡
                if self.useOcc:
                    imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(imgMap))
                    imgMap = proc.image_np2tensor(imgMap)
            # endregion
            # region 3.2.3 数据处理
            # 图像RGB通道去均值（C*H*W）
            imgMap = proc.image_colorNorm(imgMap, self.means, self.stds)
            # 生成kpsMap对应的heatmap
            kpsHeatmap, kpsMap = proc.kps_heatmap(kpsMap, imgMap.shape, self.inpRes, self.outRes)
            kpsWeight = kpsMap[:, 2].clone()
            warpmat = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes])
            # endregion
            # region 3.2.4 数据存储
            imgMap_array[idx], kpsMap_array[idx], cenMap_array[idx] = imgMap, kpsMap, cenMap
            scale_array[idx], angle_array[idx], isflip_array[idx] = scale, angle, isflip
            kpsHeatmap_array.append(kpsHeatmap)
            kpsWeight_array.append(kpsWeight)
            warpmat_array.append(warpmat)
            # endregion
        imgMap_ori = proc.image_colorNorm(imgMap_ori, self.means, self.stds)
        # endregion
        # endregion
        # region 4.输出数据组织
        # angle_ori, scale_ori = torch.tensor(0.), torch.tensor(self.inpRes / 200.0)
        # cenMap_ori = torch.tensor(proc.center_calculate(img))
        meta = {"imageID": annObj["imageID"], "imagePath": annObj["imagePath"], "islabeled": annObj["islabeled"],
                "center": cenMap_array, "scale": scale_array, "isflip": isflip_array,
                "kpsMap": kpsMap_array, 'kpsWeight': kpsWeight_array, "warpmat": warpmat_array, "kpsMap_test": kpsMap_test,
                "imgMap_ori": imgMap_ori, "kpsMap_ori": kpsMap_ori, "kpsWeight_ori": kpsWeight_ori, "angle_ori": angle_ori, "scale_ori": scale_ori, "center_ori": cenMap_ori}
        # endregion
        return imgMap_array, kpsHeatmap_array, meta

    def __len__(self):
        return len(self.dataArray)

    def _draw_testImage(self, stepID, imgMap, kpsMap, id, isflip, scale, angle):
        img_draw = proc.image_tensor2np(imgMap.detach().cpu().data * 255).astype(np.uint8)
        kps_draw = kpsMap.detach().cpu().data.numpy().astype(int).tolist()
        for kIdx, kp in enumerate(kps_draw):
            if kp[2] > 0:
                img_draw = proc.draw_point(img_draw, kp[0:2], radius=3, thickness=-1, color=(0, 95, 191))
        proc.image_save(img_draw, "{}/draw/dataset/{}/{}_{}_f{}_s{}_r{}.{}".format(self.basePath, self.dsType, id, stepID, 1 if isflip else 0, format(scale, ".1f"), format(angle, ".1f"), self.imgType))
        del img_draw, kps_draw
