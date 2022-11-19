# -*- coding: utf-8 -*-
import copy
import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.udaap.utils_augment import Augment


class CommDataset(torch.utils.data.Dataset):
    def __init__(self, dsType, dataArray, means, stds, isAug=False, isDraw=False, **kwargs):
        self.dsType, self.dataArray, self.means, self.stds, self.isAug, self.isDraw = dsType, copy.deepcopy(dataArray), means, stds, isAug, isDraw
        self.inpRes, self.outRes, self.basePath, self.imgType = kwargs['inpRes'], kwargs['outRes'], kwargs['basePath'], kwargs['imgType']
        self.useFlip, self.useOcc, self.sf, self.rf = kwargs['useFlip'], kwargs['useOcclusion'], kwargs['scaleRange'], kwargs['rotRange']
        if self.isAug and self.useOcc:
            self.augmentor = Augment(num_occluder=kwargs['numOccluder'])  # 从voc2012数据集中拿到一些mask形状，来遮挡图像。

    def __getitem__(self, idx):
        # region 1.数据组织
        # region 1.1 数据预处理（含图像resize）
        sf, rf = self.sf, self.rf
        angle, scale = torch.tensor(0.), torch.tensor(self.inpRes / 200.0)
        annObj = self.dataArray[idx]
        img_ori, kps_ori, kps_ori_test, dataID = proc.image_load(annObj["imagePath"]), annObj["kps"], annObj["kps_test"], annObj["id"]  # H*W*C
        imgMap_ori, kpsMap_ori_test = proc.image_np2tensor(img_ori), torch.from_numpy(np.array(kps_ori_test).astype(np.float32)) # 测试用
        img, kps, _ = proc.image_resize(img_ori, kps_ori, self.inpRes)  # H*W*C
        _, kps_test, _ = proc.image_resize(img_ori, kps_ori_test, self.inpRes)  # 测试用。用于验证选用的伪标签的质量。
        # endregion
        # region 1.2 数据组织
        cenMap = torch.tensor(proc.center_calculate(img))
        imgMap = proc.image_np2tensor(img)  # H*W*C ==> C*H*W
        kpsMap = torch.from_numpy(np.array(kps).astype(np.float32))
        kpsMap_test = torch.from_numpy(np.array(kps_test).astype(np.float32))  # 测试用。用于验证选用的伪标签的质量。
        kpsWeight_test = kpsMap_test[:, 2].clone()
        imgMap_256, kpsMap_256_test = imgMap.clone(), kpsMap_test.clone()  # 测试用
        # endregion
        # endregion
        # region 2.数据增强
        isflip = False
        if self.isAug:
            # 随机水平翻转。forward：fliplr==>rotation(angle)；因此，反向操作应该是：rotation(-angle)==>fliplr_back
            if self.useFlip:
                imgMap, kpsMap, cenMap, isflip = aug.fliplr(imgMap, kpsMap, cenMap, prob=0.5)
                imgMap_flip, kpsMap_flip = imgMap.clone(), kpsMap.clone()  # 测试用
            # 随机加噪（随机比例去均值）
            imgMap = aug.noisy_mean(imgMap)
            # 随机仿射变换（随机缩放、随机旋转）
            imgMap, kpsMap, scale, angle = aug.affine(imgMap, kpsMap, cenMap, scale, sf, angle, rf, [self.inpRes, self.inpRes])
            imgMap_affine, kpsMap_affine = imgMap.clone(), kpsMap.clone()  # 测试用
            # 随机遮挡
            if self.useOcc:
                imgMap, _ = self.augmentor.augment_occlu(proc.image_tensor2np(imgMap))
                imgMap = proc.image_np2tensor(imgMap)
                imgMap_occlusion = imgMap.clone()  # 测试用
        # endregion
        # region 3.数据处理
        # 图像RGB通道去均值（C*H*W）
        imgMap = proc.image_colorNorm(imgMap, self.means, self.stds)
        # 生成关键点heatmap
        kpsHeatmap, kpsMap = proc.kps_heatmap(kpsMap, imgMap.shape, self.inpRes, self.outRes)
        kpsWeight = kpsMap[:, 2].clone()
        # endregion
        # region 4.数据增强测试
        if self.isDraw:
            # 输出原图（含关键点）
            self._draw_testImage("01_ori", imgMap_ori, kpsMap_ori_test, dataID, isflip, scale, angle)
            # 输出resize图（256）
            self._draw_testImage("02_256", imgMap_256, kpsMap_256_test, dataID, isflip, scale, angle)
            # 输出水平翻转图
            if self.isAug and self.useFlip:
                self._draw_testImage("03_flip", imgMap_flip, kpsMap_flip, dataID, isflip, scale, angle)
            # 输出变换后图（含关键点）
            if self.isAug:
                self._draw_testImage("04_affine", imgMap_affine, kpsMap_affine, dataID, isflip, scale, angle)
            # 输出遮挡后图（含关键点）
            if self.isAug and self.useOcc:
                self._draw_testImage("05_occlus", imgMap_occlusion, kpsMap_affine, dataID, isflip, scale, angle)
            # 输出warpmat变换后图（含关键点）
            if annObj["islabeled"] > 0:
                kpsHeatmap_draw = kpsHeatmap.clone().unsqueeze(0)
                # 进行反向仿射变换
                warpmat = aug.affine_getWarpmat(-angle, 1, matrixRes=[self.inpRes, self.inpRes]).unsqueeze(0)
                affine_grid = F.affine_grid(warpmat, kpsHeatmap_draw.size(), align_corners=True)
                kpsHeatmap_draw = F.grid_sample(kpsHeatmap_draw, affine_grid, align_corners=True).squeeze(0)
                # 进行反向水平翻转
                if isflip: kpsHeatmap_draw = aug.fliplr_back(kpsHeatmap_draw.detach().cpu().data)
                # 从heatmap中获得关键点
                kpsMap_warpmat = torch.ones((kpsWeight.size(0), 3))
                kpsMap_warpmat[:, 0:2] = proc.kps_fromHeatmap(kpsHeatmap_draw, cenMap, scale, [self.outRes, self.outRes], mode="single")
                kpsMap_warpmat[:, 2] = kpsWeight
                self._draw_testImage("06_warpmat", imgMap_256, kpsMap_warpmat, dataID, isflip, scale, angle)
            # 输出最终图（含关键点）
            if annObj["islabeled"] > 0:
                self._draw_testImage("07_final", imgMap, kpsMap, dataID, isflip, scale, angle)
            else:
                self._draw_testImage("07_final_test", imgMap, kpsMap_test, dataID, isflip, scale, angle)
        # endregion
        # region 5.数据修正
        # kpsMap = kpsMap
        meta = {"imageID": annObj["imageID"], "imagePath": annObj["imagePath"], "center": cenMap, "scale": scale,
                "kpsMap": kpsMap, 'kpsWeight': kpsWeight, "kpsMap_test": kpsMap_test}
        # endregion
        return imgMap, kpsHeatmap, meta

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

