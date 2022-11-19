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

import cv2


class MultiDataset(torch.utils.data.Dataset):
    def update(self, pseudoArrays):
        self.dataArray = copy.deepcopy(self.dataArray_reset)
        for paIdx, pseudoArray in enumerate(pseudoArrays):
            for pseudoItem in pseudoArray:
                if pseudoItem["enable"] > 0:
                    kIdx = pseudoItem["kpID"].split("_")[-1]
                    kIdx_len = len(kIdx)
                    imageID = pseudoItem["kpID"][0:(-1-kIdx_len)]
                    # imageID, kIdx = pseudoItem["kpID"].split("_")[-1]
                    dataItem = [item for item in self.dataArray if item["imageID"] == imageID][0]
                    dataItem["kps"][paIdx][int(kIdx)] = [pseudoItem["coord"][0], pseudoItem["coord"][1], pseudoItem["enable"]]
                    dataItem["islabeled"][paIdx] = 1

    def __init__(self, dsType, dataArray, means, stds, mdsCount=2, multiCount=2, isAug=False, isDraw=False, **kwargs):
        self.mdsCount, self.multiCount = mdsCount, multiCount
        self.dataArray = self._dataArray_init(dataArray, mdsCount)
        self.dataArray_reset = copy.deepcopy(self.dataArray)
        self.dsType, self.means, self.stds, self.isAug, self.isDraw = dsType, means, stds, isAug, isDraw
        self.inpRes, self.outRes, self.basePath, self.imgType = kwargs['inpRes'], kwargs['outRes'], kwargs['basePath'], kwargs['imgType']
        self.useFlip, self.useOcc, self.sf, self.rf = kwargs['useFlip'], kwargs['useOcclusion'], kwargs['scaleRange'], kwargs['rotRange']
        self.useOcc_ema, self.sf_ema, self.rf_ema = kwargs['useOcclusion_ema'], kwargs['scaleRange_ema'], kwargs['rotRange_ema']
        if self.isAug:
            if self.useOcc:
                self.augmentor = Augment(num_occluder=kwargs['numOccluder'])  # student专用。从voc2012数据集中拿到一些mask形状，来遮挡图像。
            if self.useOcc_ema:
                self.augmentor_ema = Augment(num_occluder=kwargs['numOccluder_ema'])  # teacher专用。从voc2012数据集中拿到一些mask形状，来遮挡图像。

    def __getitem__(self, idx):
        # region 1.多通道数据准备
        # region 1.1 数据预处理（含图像resize）
        annObj = self.dataArray[idx]
        img_ori, kpsArray_ori, kps_ori_test, dataID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["id"]  # H*W*C
        img, kpsArray, _ = proc.image_resize_mulKps(img_ori, kpsArray_ori, self.inpRes)  # H*W*C
        _, kps_test, _ = proc.image_resize(img_ori, kps_ori_test, self.inpRes)  # 测试用。用于验证选用的伪标签的质量。
        # endregion
        # region 1.2 数据组织
        angle, scale = torch.tensor(0.), torch.tensor(self.inpRes / 200.0)
        imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        kpsMapArray = [torch.from_numpy(np.array(kps).astype(np.float32)) for kps in kpsArray]
        kpsMap_test = torch.from_numpy(np.array(kps_test).astype(np.float32))  # 测试用。用于验证选用的伪标签的质量。
        cenMap = torch.tensor(proc.center_calculate(img))
        imgMap_256 = imgMap.clone()  # 测试用
        # endregion
        # region 1.3 数据准备
        angle_array, scale_array, imgMap_array, kpsMapArrays, cenMap_array, isflip_array = [angle], [scale], [imgMap], [kpsMapArray], [cenMap], [False]
        for mulIdx in range(self.multiCount-1):
            angle_array.append(angle.clone())
            scale_array.append(scale.clone())
            imgMap_array.append(imgMap.clone())
            kpsMapArrays.append([kpsMap.clone() for kpsMap in kpsMapArrays[0]])
            cenMap_array.append(cenMap.clone())
            isflip_array.append(False)
        # endregion
        # endregion
        # region 2.多通道数据处理
        kpsHeatmapArrays, kpsWeightArrays, warpmatArray = [],[],[]
        for idx in range(self.multiCount):
            # region 2.1 数据准备
            imgMap, kpsMapArray, cenMap = imgMap_array[idx], kpsMapArrays[idx], cenMap_array[idx]
            scale, angle, isflip = scale_array[idx], angle_array[idx], isflip_array[idx]
            # endregion
            # region 2.2 数据增强
            if self.isAug:
                # 随机水平翻转
                imgMap, kpsMapArray, cenMap, isflip = aug.fliplr_mulKps(imgMap, kpsMapArray, cenMap, prob=0.5)
                # 随机加噪（随机比例去均值）
                imgMap = aug.noisy_mean(imgMap)
                # 随机仿射变换（随机缩放、随机旋转）
                imgMap, kpsMapArray, scale, angle = aug.affine_mulKps(imgMap, kpsMapArray, cenMap, scale, self.sf, angle, self.rf, [self.inpRes, self.inpRes])
                # 随机遮挡
                if self.useOcc:
                    imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(imgMap))
                    imgMap = proc.image_np2tensor(imgMap)
            # endregion
            # region 2.3 数据处理
            # 图像RGB通道去均值（C*H*W）
            imgMap = proc.image_colorNorm(imgMap, self.means, self.stds)
            # 生成kpsMap对应的heatmap
            kpsHeatmapArray, kpsMapArray = proc.kps_heatmap_mulKps(kpsMapArray, imgMap.shape, self.inpRes, self.outRes)
            kpsWeightArray = [kpsMap[:, 2].clone() for kpsMap in kpsMapArray]
            warpmat = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes])
            # endregion
            # region 2.4 数据归档
            imgMap_array[idx], kpsMapArrays[idx], cenMap_array[idx] = imgMap, kpsMapArray, cenMap
            scale_array[idx], angle_array[idx], isflip_array[idx] = scale, angle, isflip
            kpsHeatmapArrays.append(kpsHeatmapArray)
            kpsWeightArrays.append(kpsWeightArray)
            warpmatArray.append(warpmat)
            # endregion
            # region 2.5 数据增强测试
            # region 2.5.1 测试mds1的warpmat变换效果
            if self.isDraw and annObj["islabeled"][0] > 0:
                kpsHeatmap_draw = kpsHeatmapArray[0].clone().unsqueeze(0)
                # 进行反向仿射变换
                # warpmat_draw = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes]).unsqueeze(0)
                warpmat_draw = warpmat.unsqueeze(0)
                affine_grid = F.affine_grid(warpmat_draw, kpsHeatmap_draw.size(), align_corners=True)
                kpsHeatmap_draw = F.grid_sample(kpsHeatmap_draw, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if isflip: kpsHeatmap_draw = aug.fliplr_back(kpsHeatmap_draw.detach().cpu().data)
                # 从heatmap中获得关键点
                kpsMap_warpmat = torch.ones((kpsWeightArray[0].size(0), 3))
                kpsMap_warpmat[:, 0:2] = proc.kps_fromHeatmap(kpsHeatmap_draw, cenMap, scale, [self.outRes, self.outRes], mode="single")
                kpsMap_warpmat[:, 2] = kpsWeightArray[0]
                # self._draw_testImage("mds1", imgMap, kpsMap_warpmat, dataID, isflip, scale, angle)
                self._draw_testImage("mds1", imgMap_256.clone(), kpsMap_warpmat, dataID, isflip, scale, angle)
            # endregion

            # region 2.5.2 测试mds2的warpmat变换效果
            if self.isDraw and annObj["islabeled"][1] > 0:
                kpsHeatmap_draw = kpsHeatmapArray[1].clone().unsqueeze(0)
                # 进行反向仿射变换
                # warpmat_draw = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes]).unsqueeze(0)
                warpmat_draw = warpmat.unsqueeze(0)
                affine_grid = F.affine_grid(warpmat_draw, kpsHeatmap_draw.size(), align_corners=True)
                kpsHeatmap_draw = F.grid_sample(kpsHeatmap_draw, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if isflip: kpsHeatmap_draw = aug.fliplr_back(kpsHeatmap_draw.detach().cpu().data)
                # 从heatmap中获得关键点
                kpsMap_warpmat = torch.ones((kpsWeightArray[1].size(0), 3))
                kpsMap_warpmat[:, 0:2] = proc.kps_fromHeatmap(kpsHeatmap_draw, cenMap, scale, [self.outRes, self.outRes], mode="single")
                kpsMap_warpmat[:, 2] = kpsWeightArray[1]
                # self._draw_testImage("mds2", imgMap, kpsMap_warpmat, dataID, isflip, scale, angle)
                self._draw_testImage("mds2", imgMap_256.clone(), kpsMap_warpmat, dataID, isflip, scale, angle)
            # endregion
            # endregion
        # endregion
        # region 6.输出数据组织
        meta = {"imageID": annObj["imageID"], "imagePath": annObj["imagePath"], "islabeled": annObj["islabeled"], "imgMap_256": imgMap_256,
                "center": cenMap_array, "scale": scale_array, "isflip": isflip_array, "angle": angle_array,
                "kpsMap": kpsMapArrays, "kpsMap_test": kpsMap_test, 'kpsWeightArray': kpsWeightArrays, "warpmat": warpmatArray}
        # endregion
        return imgMap_array, kpsHeatmapArrays, meta

    def __len__(self):
        return len(self.dataArray)

    def _dataArray_init(self, dataArray, mdsCount):
        dataArray_res = copy.deepcopy(dataArray)
        for dataItem in dataArray_res:
            islabeled, kps = dataItem["islabeled"], dataItem["kps"]
            dataItem["islabeled"] = [islabeled for i in range(mdsCount)]
            dataItem["kps"] = [copy.deepcopy(kps) for i in range(mdsCount)]
        return dataArray_res

    def _draw_testImage(self, stepID, imgMap, kpsMap, id, isflip, scale, angle):
        img_draw = proc.image_tensor2np(imgMap.detach().cpu().data * 255).astype(np.uint8)
        kps_draw = kpsMap.detach().cpu().data.numpy().astype(int).tolist()
        for kIdx, kp in enumerate(kps_draw):
            if kp[2] > 0:
                img_draw = proc.draw_point(img_draw, kp[0:2], radius=3, thickness=-1, color=(0, 95, 191))
        proc.image_save(img_draw, "{}/draw/dataset/{}/{}_{}_f{}_s{}_r{}.{}".format(self.basePath, self.dsType, id, stepID, 1 if isflip else 0, format(scale, ".1f"), format(angle, ".1f"), self.imgType))
        del img_draw, kps_draw
