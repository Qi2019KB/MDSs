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


# 用于Mean-Teacher结构的数据供给。
# 特性是：数据增强后，未提供teacher的样本标签数据（既，未提供kpsHeatmap_ema）
class MTDataset(torch.utils.data.Dataset):
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

    def __init__(self, dsType, dataArray, means, stds, isAug=False, isDraw=False, **kwargs):
        self.dsType, self.dataArray, self.means, self.stds, self.isAug, self.isDraw = dsType, copy.deepcopy(dataArray), means, stds, isAug, isDraw
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
        # region 1.数据准备
        # region 1.1 数据预处理（含图像resize）
        annObj = self.dataArray[idx]
        img_ori, kps_ori, kps_ori_test, dataID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["id"]  # H*W*C
        img, kps, _ = proc.image_resize(img_ori, kps_ori, self.inpRes)  # H*W*C
        # endregion
        # region 1.2 student数据组织
        angle, scale = torch.tensor(0.), torch.tensor(self.inpRes / 200.0)
        imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        kpsMap = torch.from_numpy(np.array(kps).astype(np.float32))
        cenMap = torch.tensor(proc.center_calculate(img))
        imgMap_256 = imgMap.clone()  # 测试用
        # endregion
        # region 1.3 teacher数据组织
        angle_ema, scale_ema = angle.clone(), scale.clone()
        imgMap_ema, kpsMap_ema, cenMap_ema = imgMap.clone(), kpsMap.clone(), cenMap.clone()
        # endregion
        # endregion
        # region 2.数据增强
        isflip, isflip_ema = False, False
        if self.isAug:
            # forward：fliplr==>rotation(angle)；因此，反向操作应该是：rotation(-angle)==>fliplr_back
            # 2.1 随机水平翻转
            if self.useFlip:
                imgMap, kpsMap, cenMap, isflip = aug.fliplr(imgMap, kpsMap, cenMap, prob=0.5)
                imgMap_ema, kpsMap_ema, cenMap_ema, isflip_ema = aug.fliplr(imgMap_ema, kpsMap_ema, cenMap_ema, prob=0.5)
            # 2.2 随机加噪（随机比例去均值）
            imgMap = aug.noisy_mean(imgMap)
            imgMap_ema = aug.noisy_mean(imgMap_ema)
            # 2.3 随机仿射变换（随机缩放、随机旋转）
            imgMap, kpsMap, scale, angle = aug.affine(imgMap, kpsMap, cenMap, scale, self.sf, angle, self.rf, [self.inpRes, self.inpRes])
            imgMap_ema, kpsMap_ema, scale_ema, angle_ema = aug.affine(imgMap_ema, kpsMap_ema, cenMap_ema, scale_ema, self.sf_ema, angle_ema, self.rf_ema, [self.inpRes, self.inpRes])
            # 2.4 随机遮挡
            if self.useOcc:
                imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(imgMap))
                imgMap = proc.image_np2tensor(imgMap)
            if self.useOcc_ema:
                imgMap_ema, _ = self.augmentor_ema.augment_occlu(proc.image_tensor2np(imgMap_ema))
                imgMap_ema = proc.image_np2tensor(imgMap_ema)
        # endregion
        # region 3.数据处理
        # 图像RGB通道去均值（C*H*W）
        imgMap = proc.image_colorNorm(imgMap, self.means, self.stds)
        imgMap_ema = proc.image_colorNorm(imgMap_ema, self.means, self.stds)
        # 生成kpsMap对应的heatmap
        kpsHeatmap, kpsMap = proc.kps_heatmap(kpsMap, imgMap.shape, self.inpRes, self.outRes)
        kpsWeight = kpsMap[:, 2].clone()
        warpmat = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes])
        # 生成kpsMap_ema对应的heatmap（测试用。仅用于测试数据变换的效果）
        kpsHeatmap_ema, kpsMap_ema = proc.kps_heatmap(kpsMap_ema, imgMap_ema.shape, self.inpRes, self.outRes)
        kpsWeight_ema = kpsMap_ema[:, 2].clone()
        warpmat_ema = aug.affine_getWarpmat(-angle_ema, 1, matrixRes=[self.inpRes, self.inpRes])
        # endregion
        # region 4.数据增强测试
        if self.isDraw and annObj["islabeled"] > 0:
            # region 4.1 测试student的warpmat变换效果
            kpsHeatmap_draw = kpsHeatmap.clone().unsqueeze(0)
            # 进行反向仿射变换
            warpmat_draw = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes]).unsqueeze(0)
            affine_grid = F.affine_grid(warpmat_draw, kpsHeatmap_draw.size(), align_corners=True)
            kpsHeatmap_draw = F.grid_sample(kpsHeatmap_draw, affine_grid, align_corners=True).squeeze(0)
            # 进行反向水平翻转
            if isflip: kpsHeatmap_draw = aug.fliplr_back(kpsHeatmap_draw.detach().cpu().data)
            # 从heatmap中获得关键点
            kpsMap_warpmat = torch.ones((kpsWeight.size(0), 3))
            kpsMap_warpmat[:, 0:2] = proc.kps_fromHeatmap(kpsHeatmap_draw, cenMap, scale, [self.outRes, self.outRes], mode="single")
            kpsMap_warpmat[:, 2] = kpsWeight
            self._draw_testImage("06_warpmat", imgMap_256.clone(), kpsMap_warpmat, dataID, isflip, scale, angle)
            # endregion

            # region 4.2 测试teacher的warpmat变换效果
            kpsHeatmap_ema_draw = kpsHeatmap_ema.clone().unsqueeze(0)
            # 进行反向仿射变换
            warpmat_ema_draw = aug.affine_getWarpmat(-angle_ema, 1, matrixRes=[self.inpRes, self.inpRes]).unsqueeze(0)
            affine_ema_grid = F.affine_grid(warpmat_ema_draw, kpsHeatmap_ema_draw.size(), align_corners=True)
            kpsHeatmap_ema_draw = F.grid_sample(kpsHeatmap_ema_draw, affine_ema_grid, align_corners=True).squeeze(0)
            # 进行反向水平翻转
            if isflip_ema: kpsHeatmap_ema_draw = aug.fliplr_back(kpsHeatmap_ema_draw.detach().cpu().data)
            # 从heatmap中获得关键点
            kpsMap_ema_warpmat = torch.ones((kpsWeight_ema.size(0), 3))
            kpsMap_ema_warpmat[:, 0:2] = proc.kps_fromHeatmap(kpsHeatmap_ema_draw, cenMap_ema, scale_ema, [self.outRes, self.outRes], mode="single")
            kpsMap_ema_warpmat[:, 2] = kpsWeight_ema
            self._draw_testImage("06_warpmat_ema", imgMap_256.clone(), kpsMap_ema_warpmat, dataID, isflip_ema, scale_ema, angle_ema)
            # endregion
        # endregion
        # region 5.数据修正
        # kpsMap = kpsMap
        # endregion
        # region 6.输出数据组织
        meta = {"islabeled": annObj["islabeled"],
                "center": cenMap, "scale": scale, "isflip": isflip, "warpmat":warpmat, "kpsMap": kpsMap, 'kpsWeight': kpsWeight,
                "center_ema": cenMap_ema, "scale_ema": scale_ema, "isflip_ema": isflip_ema, "warpmat_ema":warpmat_ema, 'kpsWeight_ema': kpsWeight_ema}
        # endregion
        return imgMap, kpsHeatmap, imgMap_ema, meta

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
