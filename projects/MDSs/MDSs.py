# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import numpy as np
import torch
from torch.optim.adam import Adam as TorchAdam
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
import copy

import GLOB as glob
import datasources
import datasets
import models

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.parameters import consWeight_increase, pseudoWeight_increase, PALWeight_decrease, update_ema_variables
from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.evaluation import EvaluationUtils as eval
from utils.business import BusinessUtils as bus
from utils.losses import GateJointMSELoss, GateJointDistLoss, AvgCounter, AvgCounters

import matplotlib
matplotlib.use('Agg')


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args_init = copy.deepcopy(args)
    args.global_step = 0
    args.mds1_best_acc = -1.
    args.mds1_best_epoch = 0
    args.mds2_best_acc = -1.
    args.mds2_best_epoch = 0
    args.mean_best_acc = -1.
    args.mean_best_epoch = 0
    args.mds1_lma_cache, args.mds2_lma_cache = [], []
    # region 1. dataloader initialize
    # data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgType, args.pck_ref = dataSource.kpsCount, dataSource.imgType, dataSource.pck_ref

    # train-set dataloader
    trainDS = datasets.__dict__["DS_mds"]("train", semiTrainData, means, stds, multiCount=2, isAug=True, isDraw=False, **vars(args))
    trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=False, pin_memory=True, drop_last=False)
    # valid-set dataloader
    validDS = datasets.__dict__["DS"]("valid", validData, means, stds, isAug=False, isDraw=True, **vars(args))
    validLoader = TorchDataLoader(validDS, batch_size=args.inferBS, shuffle=False, pin_memory=True, drop_last=False)
    # pseudo-labels dataloader
    pseudoDS = datasets.__dict__["DS_multi"]("pseudo", unlabeledData, means, stds, multiCount=2*args.augSampleNum, isAug=True, isDraw=False, **vars(args))
    pseudoLoader = TorchDataLoader(pseudoDS, batch_size=args.inferBS, shuffle=False, pin_memory=True, drop_last=False)
    logger.print("L1", "=> initialized {} Dataset loaders".format(args.dataSource))
    # endregion

    # region 2. modules initialize
    # Mean-Teacher Module 1
    mds1_model = models.__dict__["HG"](args.kpsCount, args.nStack).to(args.device)
    mds1_model_ema = models.__dict__["HG"](args.kpsCount, args.nStack, nograd=True).to(args.device)
    mds1_optim = TorchAdam(mds1_model.parameters(), lr=args.lr, weight_decay=args.wd)
    # Mean-Teacher Module 2
    mds2_model = models.__dict__["HG"](args.kpsCount, args.nStack).to(args.device)
    mds2_model_ema = models.__dict__["HG"](args.kpsCount, args.nStack, nograd=True).to(args.device)
    mds2_optim = TorchAdam(mds2_model.parameters(), lr=args.lr, weight_decay=args.wd)
    hg_pNum = sum(p.numel() for p in mds1_model.parameters())
    logc = "=> initialized MDSs models (nStack: {}, params: {})".format(args.nStack, format(hg_pNum*4 / 1000000.0, ".2f"))
    logger.print("L1", logc)
    # endregion

    # region 3. iteration
    logger.print("L1", "=> training start")
    simModel_s1t1, simModel_s2t2, simModel_s1s2, simModel_t1t2 = [], [], [], []
    simVal_t1t2 = []
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()

        # region 3.1 update dynamic parameters
        args.pseudoWeight = pseudoWeight_increase(epo, args)
        args.consWeight = consWeight_increase(epo, args)
        args.PALWeight = args.PALWeight_max  # PALWeight_decrease(epo, args)
        # endregion

        # region 3.2 model training and validating
        trainTM_start = datetime.datetime.now()
        [mds1_icc_loss, mds1_ecc_loss, mds1_pec_loss], [mds2_icc_loss, mds2_ecc_loss, mds2_pec_loss], pac_loss = train(trainLoader, mds1_model, mds1_model_ema, mds1_optim, mds2_model, mds2_model_ema, mds2_optim, args)
        preds_mds1_ema, mds1_ema_accs, mds1_ema_errs, preds_mds2_ema, mds2_ema_accs, mds2_ema_errs, preds_mean, mean_accs, mean_errs = validate(validLoader, mds1_model_ema, mds2_model_ema, args, epo)
        trainTM_end = datetime.datetime.now()
        # endregion

        # region 3.3 model selection & storage
        # region 3.3.1 mds1
        # model selection
        mds1_is_best = mds1_ema_accs[-1] > args.mds1_best_acc
        if mds1_is_best:
            args.mds1_best_epoch = epo + 1
            args.mds1_best_acc = mds1_ema_accs[-1]
        # model storage
        comm.ckpt_save({
            'epoch': epo + 1,
            'model': "HG",
            'global_step': args.global_step,
            'best_acc': args.mds1_best_acc,
            'best_epoch': args.mds1_best_epoch,
            'state_dict': mds1_model.state_dict(),
            'state_dict_ema': mds1_model_ema.state_dict(),
            'optim': mds1_optim.state_dict()
        }, mds1_is_best, ckptPath="{}/ckpts/mds1".format(args.basePath))
        # endregion
        # region 3.3.2 mds2
        # model selection
        mds2_is_best = mds2_ema_accs[-1] > args.mds2_best_acc
        if mds2_is_best:
            args.mds2_best_epoch = epo + 1
            args.mds2_best_acc = mds2_ema_accs[-1]
        # model storage
        comm.ckpt_save({
            'epoch': epo + 1,
            'model': "HG",
            'global_step': args.global_step,
            'best_acc': args.mds2_best_acc,
            'best_epoch': args.mds2_best_epoch,
            'state_dict': mds2_model.state_dict(),
            'state_dict_ema': mds2_model_ema.state_dict(),
            'optim': mds2_optim.state_dict()
        }, mds2_is_best, ckptPath="{}/ckpts/mds2".format(args.basePath))
        # endregion
        # region 3.3.3 mean
        # model selection
        mean_is_best = mean_accs[-1] > args.mean_best_acc
        if mean_is_best:
            args.mean_best_epoch = epo + 1
            args.mean_best_acc = mean_accs[-1]
        # endregion
        # endregion

        # region 3.4 create pseudo-labels
        pseudoTM_start = datetime.datetime.now()
        # create pseudo-labels
        mds1_pseudoArray, mds1_selCounts, mds1_selErrs, mds1_selAccs, mds1_uncThr, mds2_pseudoArray, mds2_selCounts, mds2_selErrs, mds2_selAccs, mds2_uncThr = infer(pseudoLoader, mds1_model_ema, mds2_model_ema, args)
        # update dataset
        trainDS.update([mds1_pseudoArray, mds2_pseudoArray])
        trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=False, pin_memory=True, drop_last=False)
        pseudoTM_end = datetime.datetime.now()
        # endregion

        # region 3.5 model evaluation
        simTM_start = datetime.datetime.now()
        # similarity evaluate
        simModel_s1t1.append(eval.modelSimilarity_byCosineSimilarity(mds1_model, mds1_model_ema))
        simModel_s2t2.append(eval.modelSimilarity_byCosineSimilarity(mds2_model, mds2_model_ema))
        simModel_s1s2.append(eval.modelSimilarity_byCosineSimilarity(mds1_model, mds2_model))
        simModel_t1t2.append(eval.modelSimilarity_byCosineSimilarity(mds1_model_ema, mds2_model_ema))
        simVal_t1t2.append(eval.predsSimilarity_byDistance(preds_mds1_ema, preds_mds2_ema))

        # draw line chart
        if (epo+1) % 5 == 0 or epo == args.epochs - 1:  # epo > 0 and epo % 5 == 0
            statistics_savePath = "{}/statistics".format(args.basePath)
            if not os.path.exists(statistics_savePath): os.makedirs(statistics_savePath)

            # 绘制Model Similarity曲线
            modelSim_savePath = "{}/modelSimilarity.svg".format(statistics_savePath)
            if os.path.isfile(modelSim_savePath): os.remove(modelSim_savePath)
            matplotlib.pyplot.clf()
            matplotlib.pyplot.plot(simModel_s1t1, label="mds1stu vs mds1tea", linestyle=":", linewidth=1, color="k")
            matplotlib.pyplot.plot(simModel_s2t2, label="mds2stu vs mds2tea", linestyle=":", linewidth=1, color="g")
            matplotlib.pyplot.legend()
            matplotlib.pyplot.xlabel('Epochs', fontsize=15)
            matplotlib.pyplot.ylabel('Similarity', fontsize=15)
            matplotlib.pyplot.title("Model Similarity")
            matplotlib.pyplot.savefig(modelSim_savePath, bbox_inches='tight')

            # 绘制Model Similarity曲线
            modelSim_savePath = "{}/modelSimilarity2.svg".format(statistics_savePath)
            if os.path.isfile(modelSim_savePath): os.remove(modelSim_savePath)
            matplotlib.pyplot.clf()
            matplotlib.pyplot.plot(simModel_s1s2, label="mds1stu vs mds2stu", linestyle="-", linewidth=1, color="y")
            matplotlib.pyplot.plot(simModel_t1t2, label="mds1tea vs mds2tea", linestyle="-", linewidth=1, color="b")
            matplotlib.pyplot.legend()
            matplotlib.pyplot.xlabel('Epochs', fontsize=15)
            matplotlib.pyplot.ylabel('Similarity', fontsize=15)
            matplotlib.pyplot.title("Model Similarity")
            matplotlib.pyplot.savefig(modelSim_savePath, bbox_inches='tight')

            # 绘制Prediction Similarity曲线
            predsSim_savePath = "{}/predsSimilarity.svg".format(statistics_savePath)
            if os.path.isfile(predsSim_savePath): os.remove(predsSim_savePath)
            matplotlib.pyplot.clf()
            matplotlib.pyplot.plot(simVal_t1t2, label="mds1tea vs mds2tea", linestyle="-", linewidth=1, color="b")
            matplotlib.pyplot.legend()
            matplotlib.pyplot.xlabel('Epochs', fontsize=15)
            matplotlib.pyplot.ylabel('Similarity', fontsize=15)
            matplotlib.pyplot.title("Prediction Similarity")
            # matplotlib.pyplot.ylim(ymin=0.0, ymax=1)
            matplotlib.pyplot.savefig(predsSim_savePath, bbox_inches='tight')
        simTM_end = datetime.datetime.now()
        # endregion

        # region 3.6 log storage
        log_dataItem = {"mean_errs": mean_errs, "mean_accs": mean_accs,
                              "pac_loss": pac_loss, "simModel_s1t1": simModel_s1t1, "simModel_s2t2": simModel_s2t2,
                              "simModel_s1s2": simModel_s1s2, "simModel_t1t2": simModel_t1t2,
                              "simVal_t1t2": simVal_t1t2,
                              "mds1_icc_loss": mds1_icc_loss, "mds1_ecc_loss": mds1_ecc_loss,
                              "mds1_pec_loss": mds1_pec_loss, "mds1_ema_errs": mds1_ema_errs,
                              "mds1_ema_accs": mds1_ema_accs,
                              "mds1_pseudoArray": mds1_pseudoArray, "mds1_selCounts": mds1_selCounts,
                              "mds1_selErrs": mds1_selErrs, "mds1_selAccs": mds1_selAccs, "mds1_uncThr": mds1_uncThr,
                              "mds2_icc_loss": mds2_icc_loss, "mds2_ecc_loss": mds2_ecc_loss,
                              "mds2_pec_loss": mds2_pec_loss, "mds2_ema_errs": mds2_ema_errs,
                              "mds2_ema_accs": mds2_ema_accs,
                              "mds2_pseudoArray": mds2_pseudoArray, "mds2_selCounts": mds2_selCounts,
                              "mds2_selErrs": mds2_selErrs, "mds2_selAccs": mds2_selAccs, "mds2_uncThr": mds2_uncThr}
        comm.json_save([log_dataItem], "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)
        if epo == 0:
            save_args = vars(args_init).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)
        # endregion

        # region 3.7 output result
        fmtc = "[{}/{} | mds1 | consW: {}, PALW: {}] best acc: {} (epo: {}) | err: {}, acc: {} | icc_loss: {}, ecc_loss: {}, pec_loss: {}, pac_loss: {} | kps_acc:[{}], kps_err:[{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.consWeight, ".2f"), format(args.PALWeight, ".5f"),
                           format(args.mds1_best_acc, ".3f"), format(args.mds1_best_epoch, "3d"), format(mds1_ema_errs[-1], ".2f"), format(mds1_ema_accs[-1], ".3f"),
                           format(mds1_icc_loss, ".5f"), format(mds1_ecc_loss, ".5f"), format(mds1_pec_loss, ".5f"), format(pac_loss, ".8f"),
                           setContent(mds1_ema_accs[0:len(mds1_ema_accs)-1], ".3f"), setContent(mds1_ema_errs[0:len(mds1_ema_errs)-1], ".2f"))
        logger.print("L1", logc, start=trainTM_start, end=trainTM_end)
        fmtc = "[{}/{} | mds2 | consW: {}, PALW: {}] best acc: {} (epo: {}) | err: {}, acc: {} | icc_loss: {}, ecc_loss: {}, pec_loss: {}, pac_loss: {} | kps_acc:[{}], kps_err:[{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.consWeight, ".2f"), format(args.PALWeight, ".5f"),
                           format(args.mds2_best_acc, ".3f"), format(args.mds2_best_epoch, "3d"), format(mds2_ema_errs[-1], ".2f"), format(mds2_ema_accs[-1], ".3f"),
                           format(mds2_icc_loss, ".5f"), format(mds2_ecc_loss, ".5f"), format(mds2_pec_loss, ".5f"), format(pac_loss, ".8f"),
                           setContent(mds2_ema_accs[0:len(mds2_ema_accs)-1], ".3f"), setContent(mds2_ema_errs[0:len(mds2_ema_errs)-1], ".2f"))
        logger.print("L1", logc, start=trainTM_start, end=trainTM_end)
        fmtc = "[{}/{} | mean | consW: {}, PALW: {}] best acc: {} (epo: {}) | err: {}, acc: {} | kps_acc:[{}], kps_err:[{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.consWeight, ".2f"), format(args.PALWeight, ".5f"),
                           format(args.mean_best_acc, ".3f"), format(args.mean_best_epoch, "3d"), format(mean_errs[-1], ".2f"), format(mean_accs[-1], ".3f"),
                           setContent(mean_accs[0:len(mean_accs)-1], ".3f"), setContent(mean_errs[0:len(mean_errs)-1], ".2f"))
        logger.print("L1", logc, start=trainTM_start, end=trainTM_end)
        fmtc = "[{}/{} | mds1's pseudoW: {}, uncThr: {}] selCount: {}|{}, [{}]; selAcc: {}, [{}]; selError: {}, [{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.pseudoWeight, ".3f"), format(mds2_uncThr, ".2f"),
                           format(mds1_selCounts[-1], "1d"), format(len(mds1_pseudoArray), "1d"), setContent(mds1_selCounts[0:len(mds1_selCounts) - 1], "1d"),
                           format(mds1_selAccs[-1], ".2f"), setContent(mds1_selAccs[0:len(mds1_selAccs) - 1], ".2f"),
                           format(mds1_selErrs[-1], ".2f"), setContent(mds1_selErrs[0:len(mds1_selErrs) - 1], ".2f"))
        logger.print("L1", logc, start=pseudoTM_start, end=pseudoTM_end)
        fmtc = "[{}/{} | mds2's pseudoW: {}, uncThr: {}] selCount: {}|{}, [{}]; selAcc: {}, [{}]; selError: {}, [{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.pseudoWeight, ".3f"), format(mds2_uncThr, ".2f"),
                           format(mds2_selCounts[-1], "1d"), format(len(mds2_pseudoArray), "1d"), setContent(mds2_selCounts[0:len(mds2_selCounts) - 1], "1d"),
                           format(mds2_selAccs[-1], ".2f"), setContent(mds2_selAccs[0:len(mds2_selAccs) - 1], ".2f"),
                           format(mds2_selErrs[-1], ".2f"), setContent(mds2_selErrs[0:len(mds2_selErrs) - 1], ".2f"))
        logger.print("L1", logc, start=pseudoTM_start, end=pseudoTM_end)
        fmtc = "[{}/{} | distance] ModelDist: [s1t1: {}, s2t2: {}, s1s2: {}, t1t2: {}] | predsDist: [t1t2: {}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"),
                           format(simModel_s1t1[-1], ".6f"), format(simModel_s2t2[-1], ".6f"), format(simModel_s1s2[-1], ".6f"), format(simModel_t1t2[-1], ".6f"),
                           format(simVal_t1t2[-1], ".1f"))
        logger.print("L1", logc, start=simTM_start, end=simTM_end)
        fmtc = "[{}/{} | finished] ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"))
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, All executing finished...]".format(args.experiment), start=allTM)


def infer(pseudoLoader, mds1_model_ema, mds2_model_ema, args):
    mds1_model_ema.eval()
    mds2_model_ema.eval()
    with torch.no_grad():
        all_pseudoArray_mds1, all_pseudoArray_mds2 = [], []
        # region 1. create pseudo-labels
        for bat, (imgMap_array, kpsHeatmap_array, meta) in enumerate(pseudoLoader):
            imgID, kpsMap_test = meta["imageID"], meta['kpsMap_test']
            imgMap_array = [item.to(args.device, non_blocking=True) for item in imgMap_array]
            imgMap_ori = meta["imgMap_ori"].to(args.device, non_blocking=True)
            # model forward（20220818-每个tea都对5个数据增广做预测。取前3个接近的计算mean、var；如何计算：C53，取方差最小的3人组；记得仿射变换，换回标准坐标系下的坐标值）
            aug_predsArray_t1, aug_predsArray_t2 = [], []
            aug_scoresArray_t1, aug_scoresArray_t2 = [], []
            for idx in range(args.augSampleNum):
                bs, k, _, _ = kpsHeatmap_array[idx].shape
                # mds1_tea:
                aug_imgMap_t1, aug_warpmat_t1 = imgMap_array[idx*2], meta['warpmat'][idx*2].to(args.device, non_blocking=True)
                aug_center_t1, aug_scale_t1, aug_isflip_t1 = meta['center'][idx*2], meta['scale'][idx*2], meta['isflip'][idx*2]
                aug_outs_t1 = aug.affine_back(mds1_model_ema(aug_imgMap_t1)[:, -1], aug_warpmat_t1, aug_isflip_t1).to(args.device, non_blocking=True).cpu()
                aug_preds_t1, aug_scores_t1 = proc.kps_fromHeatmap(aug_outs_t1, aug_center_t1, aug_scale_t1, [args.outRes, args.outRes])  # [bs, k, 2]
                aug_predsArray_t1.append(aug_preds_t1)
                aug_scoresArray_t1.append(aug_scores_t1)

                # mds2_tea
                aug_imgMap_t2, aug_warpmat_t2 = imgMap_array[idx*2+1], meta['warpmat'][idx*2+1].to(args.device, non_blocking=True)
                aug_center_t2, aug_scale_t2, aug_isflip_t2 = meta['center'][idx*2+1], meta['scale'][idx*2+1], meta['isflip'][idx*2+1]
                aug_outs_t2 = aug.affine_back(mds2_model_ema(aug_imgMap_t2)[:, -1], aug_warpmat_t2, aug_isflip_t2).to(args.device, non_blocking=True).cpu()
                aug_preds_t2, aug_scores_t2 = proc.kps_fromHeatmap(aug_outs_t2, aug_center_t2, aug_scale_t2, [args.outRes, args.outRes])
                aug_predsArray_t2.append(aug_preds_t2)
                aug_scoresArray_t2.append(aug_scores_t2)

            # original imgMap forward
            preds_mds1, scores_mds1 = proc.kps_fromHeatmap(mds1_model_ema(imgMap_ori)[:, -1].cpu(), meta['center_ori'], meta['scale_ori'], [args.outRes, args.outRes])
            preds_mds2, scores_mds2 = proc.kps_fromHeatmap(mds2_model_ema(imgMap_ori)[:, -1].cpu(), meta['center_ori'], meta['scale_ori'], [args.outRes, args.outRes])
            aug_predsArray_mds1, aug_scoresArray_mds1 = torch.stack(aug_predsArray_t1, dim=2), torch.stack(aug_scoresArray_t1, dim=2)
            aug_predsArray_mds2, aug_scoresArray_mds2 = torch.stack(aug_predsArray_t2, dim=2), torch.stack(aug_scoresArray_t2, dim=2)
            pseudoArray_mds1, pseudoArray_mds2 = bus.pseudo_cal_unc(imgID, kpsMap_test, preds_mds1, scores_mds1, aug_predsArray_mds1, aug_scoresArray_mds1, preds_mds2, scores_mds2, aug_predsArray_mds2, aug_scoresArray_mds2, args)
            # add the pseudo-label to each other
            all_pseudoArray_mds1 += pseudoArray_mds2
            all_pseudoArray_mds2 += pseudoArray_mds1
        # endregion
        # region 2. pseudo-labels filter
        selArray_mds1, selCounts_mds1, selErrs_mds1, selAccs_mds1, uncThr_mds1 = bus.pseudo_filter_mixUnc(all_pseudoArray_mds1, args)
        selArray_mds2, selCounts_mds2, selErrs_mds2, selAccs_mds2, uncThr_mds2 = bus.pseudo_filter_mixUnc(all_pseudoArray_mds2, args)
        # endregion
    return selArray_mds1, selCounts_mds1, selErrs_mds1, selAccs_mds1, uncThr_mds1, selArray_mds2, selCounts_mds2, selErrs_mds2, selAccs_mds2, uncThr_mds2


def train(trainLoader, mds1_model, mds1_model_ema, mds1_optim, mds2_model, mds2_model_ema, mds2_optim, args):
    mds1_icc_counter, mds2_icc_counter = AvgCounter(), AvgCounter()
    mds1_ecc_counter, mds2_ecc_counter = AvgCounter(), AvgCounter()
    mds1_pec_counter, mds2_pec_counter = AvgCounter(), AvgCounter()
    pac_counter = AvgCounter()
    pose_lossFunc = GateJointMSELoss(nStack=args.nStack, useKPsGate=True, useSampleWeight=True).to(args.device)
    consistency_lossFunc = GateJointDistLoss().to(args.device)
    mds1_model.train()
    mds1_model_ema.train()
    mds2_model.train()
    mds2_model_ema.train()
    for bat, (imgMapArray, kpsHeatmapArrays, meta) in enumerate(trainLoader):
        mds1_optim.zero_grad()
        mds2_optim.zero_grad()

        # region 1. data organize
        kpsHeatmapArrays = [[setVariable(kpsHeatmap, args.device) for kpsHeatmap in kpsHeatmapArray] for kpsHeatmapArray in kpsHeatmapArrays]  # [2, 2] * [bs, 9, 256, 256]
        kpsGateArrays = [[setVariable(kpsWeight, args.device) for kpsWeight in kpsWeightArray] for kpsWeightArray in meta['kpsWeightArray']]   # [2, 2] * [bs, 9]
        imgMapArray = [setVariable(imgMap, args.device) for imgMap in imgMapArray]  # 2 * [bs, 3, 256, 256]
        warpmatArray = [setVariable(warpmat, args.device) for warpmat in meta['warpmat']]  # 2 * [bs, 2, 3]
        isflipArray = meta['isflip']  # 2 * [bs]
        sampleWeightArray = calSampleWeight_train(meta['islabeled'], args)  # 2 * [bs]; calculate the weight of the gt-sample (w = 1.0) and pseudo-label (w = args.cur_pw)
        # endregion

        # region 2. model forward
        outs_mds1, outs_mds1_ema, outs_mds2, outs_mds2_ema = [], [], [], []
        for imgMap in imgMapArray:
            outs_mds1.append(mds1_model(imgMap))
            outs_mds1_ema.append(mds1_model_ema(imgMap))
            outs_mds2.append(mds2_model(imgMap))
            outs_mds2_ema.append(mds2_model_ema(imgMap))
        # endregion

        # region 3. internal consistency constraint (model consistency)  -- [:, -1]
        mds1_icc_loss, mds2_icc_loss = 0., 0.
        # region 3.1 calculate mds1_icc_loss
        mds1_icc_sum, mds1_icc_count = 0., 0
        # calculate mds1_icc_model_loss
        preds_v1 = aug.affine_back(outs_mds1[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds1[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds1_icc_sum += loss
        mds1_icc_count += n
        # calculate mds1_icc_model_ema_loss
        preds_v1 = aug.affine_back(outs_mds1_ema[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds1_ema[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds1_icc_sum += loss
        mds1_icc_count += n
        # cal & record the mds1_icc_loss
        mds1_icc_loss += args.consWeight * ((mds1_icc_sum / mds1_icc_count) if mds1_icc_count > 0 else mds1_icc_sum)
        mds1_icc_counter.update(mds1_icc_loss.item(), mds1_icc_count)
        # endregion
        # region 3.2 calculate mds2_icc_loss
        mds2_icc_sum, mds2_icc_count = 0., 0
        # calculate mds2_icc_model_loss
        preds_v1 = aug.affine_back(outs_mds2[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds2[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds2_icc_sum += loss
        mds2_icc_count += n
        # calculate mds2_icc_model_ema_loss
        preds_v1 = aug.affine_back(outs_mds2_ema[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds2_ema[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds2_icc_sum += loss
        mds2_icc_count += n
        # cal & record the mds2_icc_loss
        mds2_icc_loss += args.consWeight * ((mds2_icc_sum / mds2_icc_count) if mds2_icc_count > 0 else mds2_icc_sum)
        mds2_icc_counter.update(mds2_icc_loss.item(), mds2_icc_count)
        # endregion
        # endregion

        # region 4. external consistency constraint (mean-teacher consistency)  -- [:, -1]
        mds1_ecc_loss, mds2_ecc_loss = 0., 0.
        # region 4.1 calculate mds1_ecc_loss
        mds1_ecc_sum, mds1_ecc_count = 0., 0
        # calculate mds1_mt_loss (same augSample)
        preds_v1 = aug.affine_back(outs_mds1[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds1_ema[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds1_ecc_sum += loss
        mds1_ecc_count += n
        preds_v1 = aug.affine_back(outs_mds1[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds1_ema[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds1_ecc_sum += loss
        mds1_ecc_count += n
        # calculate mds1_mt_loss (different augSample)
        preds_v1 = aug.affine_back(outs_mds1[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds1_ema[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds1_ecc_sum += loss
        mds1_ecc_count += n
        preds_v1 = aug.affine_back(outs_mds1[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds1_ema[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds1_ecc_sum += loss
        mds1_ecc_count += n
        # cal & record the mds1_ecc_loss
        mds1_ecc_loss += args.consWeight * ((mds1_ecc_sum / mds1_ecc_count) if mds1_ecc_count > 0 else mds1_ecc_sum)
        mds1_ecc_counter.update(mds1_ecc_loss.item(), mds1_ecc_count)
        # endregion
        # region 4.2 calculate mds2_ecc_loss
        mds2_ecc_sum, mds2_ecc_count = 0., 0
        # calculate mds2_mt_loss (same augSample)
        preds_v1 = aug.affine_back(outs_mds2[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds2_ema[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds2_ecc_sum += loss
        mds2_ecc_count += n
        preds_v1 = aug.affine_back(outs_mds2[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds2_ema[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds2_ecc_sum += loss
        mds2_ecc_count += n
        # calculate mds2_mt_loss (different augSample)
        preds_v1 = aug.affine_back(outs_mds2[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds2_ema[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds2_ecc_sum += loss
        mds2_ecc_count += n
        preds_v1 = aug.affine_back(outs_mds2[1][:, -1], warpmatArray[1], isflipArray[1]).to(args.device, non_blocking=True)
        preds_v2 = aug.affine_back(outs_mds2_ema[0][:, -1], warpmatArray[0], isflipArray[0]).to(args.device, non_blocking=True)
        loss, n = consistency_lossFunc(preds_v1, preds_v2)
        mds2_ecc_sum += loss
        mds2_ecc_count += n
        # cal & record the mds2_ecc_loss
        mds2_ecc_loss += args.consWeight * ((mds2_ecc_sum / mds2_ecc_count) if mds2_ecc_count > 0 else mds2_ecc_sum)
        mds2_ecc_counter.update(mds2_ecc_loss.item(), mds2_ecc_count)
        # endregion
        # endregion

        # region 5. pose estimation constraint
        mds1_pec_loss, mds2_pec_loss = 0., 0.
        # region 5.1 calculate mds1_pose_loss (using teacher's preds to cal the pose_loss)
        mds1_pec_sum, mds1_pec_count = 0., 0
        loss, n = pose_lossFunc(outs_mds1[0], kpsHeatmapArrays[0][0], kpsGateArrays[0][0], sampleWeightArray[0])
        mds1_pec_sum += loss
        mds1_pec_count += n
        loss, n = pose_lossFunc(outs_mds1[1], kpsHeatmapArrays[1][0], kpsGateArrays[1][0], sampleWeightArray[1])
        mds1_pec_sum += loss
        mds1_pec_count += n
        # cal & record the mds1_pec_loss
        mds1_pec_loss += args.poseWeight * ((mds1_pec_sum / mds1_pec_count) if mds1_pec_count > 0 else mds1_pec_sum)
        mds1_pec_counter.update(mds1_pec_loss.item(), mds1_pec_count)
        # endregion
        # region 5.2 calculate mds2_pose_loss (using teacher's preds to cal the pose_loss)
        mds2_pec_sum, mds2_pec_count = 0., 0
        loss, n = pose_lossFunc(outs_mds2[0], kpsHeatmapArrays[0][1], kpsGateArrays[0][1], sampleWeightArray[0])
        mds2_pec_sum += loss
        mds2_pec_count += n
        loss, n = pose_lossFunc(outs_mds2[1], kpsHeatmapArrays[1][1], kpsGateArrays[1][1], sampleWeightArray[1])
        mds2_pec_sum += loss
        mds2_pec_count += n
        # cal & record the mds1_pec_loss
        mds2_pec_loss += args.poseWeight * ((mds2_pec_sum / mds2_pec_count) if mds2_pec_count > 0 else mds2_pec_sum)
        mds2_pec_counter.update(mds2_pec_loss.item(), mds2_pec_count)
        # endregion
        # endregion

        # region 6. parameters adversarial constraint (cal the distance between mds1's student and mds2's student)
        v1, v2 = None, None
        for (p1, p2) in zip(mds1_model.parameters(), mds2_model.parameters()):
            if v1 is None and v2 is None:
                v1, v2 = p1.view(-1), p2.view(-1)
            else:
                v1, v2 = torch.cat((v1, p2.view(-1)), 0), torch.cat((v2, p2.view(-1)), 0)  # 拼接
        pac_loss = args.PALWeight * (1.0 + (torch.matmul(v1, v2) / (torch.norm(v1) * torch.norm(v2))))  # + 1  # +1 is for a positive loss
        pac_counter.update(pac_loss.item(), 1)
        # endregion

        # region 7. calculate total loss & update model
        # cal total loss
        mds1_total_loss = mds1_icc_loss + mds1_ecc_loss + mds1_pec_loss + pac_loss
        mds2_total_loss = mds2_icc_loss + mds2_ecc_loss + mds2_pec_loss + pac_loss
        # backward
        mds1_total_loss.backward(retain_graph=True)
        mds1_optim.step()
        update_ema_variables(mds1_model, mds1_model_ema, args)

        mds2_total_loss.backward()
        mds2_optim.step()
        update_ema_variables(mds2_model, mds2_model_ema, args)

        # region 8. clearing the GPU Cache
        del kpsHeatmapArrays, kpsGateArrays, imgMapArray, warpmatArray, isflipArray, sampleWeightArray
        del outs_mds1, outs_mds1_ema, outs_mds2, outs_mds2_ema
        del mds1_icc_loss, mds2_icc_loss, mds1_ecc_loss, mds2_ecc_loss, mds1_pec_loss, mds2_pec_loss, pac_loss
        del mds1_total_loss, mds2_total_loss
        # endregion
    args.global_step += 1
    return [mds1_icc_counter.avg, mds1_ecc_counter.avg, mds1_pec_counter.avg], [mds2_icc_counter.avg, mds2_ecc_counter.avg, mds2_pec_counter.avg], pac_counter.avg


def validate(validLoader, mds1_model_ema, mds2_model_ema, args, epo):
    errs_mds1_ema_counters, accs_mds1_ema_counters = AvgCounters(), AvgCounters()
    errs_mds2_ema_counters, accs_mds2_ema_counters = AvgCounters(), AvgCounters()
    errs_mean_counters, accs_mean_counters = AvgCounters(), AvgCounters()
    mds1_model_ema.eval()
    mds2_model_ema.eval()
    predsArray_mds1_ema, predsArray_mds2_ema, predsArray_mean = [], [], []
    with torch.no_grad():
        for bat, (imgMap, kpsHeatmap, meta) in enumerate(validLoader):
            # region 1. data organize
            imgMap = imgMap.to(args.device, non_blocking=True)
            bs, k, _, _ = kpsHeatmap.shape
            # endregion

            # region 2. model forward
            outs_mds1_ema = mds1_model_ema(imgMap)[:, -1].cpu()  # 多模型预测结果 [bs, nstack, k, h, w]
            outs_mds2_ema = mds2_model_ema(imgMap)[:, -1].cpu()  # 多模型预测结果 [bs, nstack, k, h, w]
            preds_mds1_ema, _ = proc.kps_fromHeatmap(outs_mds1_ema, meta['center'], meta['scale'], [args.outRes, args.outRes])  # kps_fromHeatmap(heatmap, cenMap, scale, res)
            preds_mds2_ema, _ = proc.kps_fromHeatmap(outs_mds2_ema, meta['center'], meta['scale'], [args.outRes, args.outRes])
            preds_mean = bus.preds_mean(preds_mds1_ema, preds_mds2_ema)
            predsArray_mds1_ema += preds_mds1_ema
            predsArray_mds2_ema += preds_mds2_ema
            predsArray_mean += preds_mean
            # endregion

            # region 3. calculate the error and accuracy
            errs_mds1_ema, accs_mds1_ema = eval.acc_pck(preds_mds1_ema, meta['kpsMap'], args.pck_ref, args.pck_thr)
            errs_mds2_ema, accs_mds2_ema = eval.acc_pck(preds_mds2_ema, meta['kpsMap'], args.pck_ref, args.pck_thr)
            errs_mean, accs_mean = eval.acc_pck(preds_mean, meta['kpsMap'], args.pck_ref, args.pck_thr)
            for idx in range(k+1):
                errs_mds1_ema_counters.update(idx, errs_mds1_ema[idx].item(), bs if idx < k else bs*k)  # bs*k 有问题，待调查。
                accs_mds1_ema_counters.update(idx, accs_mds1_ema[idx].item(), bs if idx < k else bs*k)
                errs_mds2_ema_counters.update(idx, errs_mds2_ema[idx].item(), bs if idx < k else bs*k)
                accs_mds2_ema_counters.update(idx, accs_mds2_ema[idx].item(), bs if idx < k else bs*k)
                errs_mean_counters.update(idx, errs_mean[idx].item(), bs if idx < k else bs*k)
                accs_mean_counters.update(idx, accs_mean[idx].item(), bs if idx < k else bs*k)
            # endregion

            # region 4. output the predictions
            if epo+1 % 100 == 0:
                for iIdx in range(bs):
                    img, imgID = proc.image_load(meta["imagePath"][iIdx]), meta["imageID"][iIdx]
                    h, w, _ = img.shape
                    gtArray = meta['kpsMap'][iIdx].cpu().data.numpy().tolist()
                    # region 4.1 draw pictures with the predictions of mds1_ema
                    predArray_mds1_ema = preds_mds1_ema[iIdx].cpu().data.numpy().tolist()
                    for (pred_256, gt_256) in zip(predArray_mds1_ema, gtArray):
                        gt = [gt_256[0]*w/args.inpRes, gt_256[1]*h/args.inpRes]
                        pred = [pred_256[0]*w/args.inpRes, pred_256[1]*h/args.inpRes]
                        img = proc.draw_point(img, gt, radius=3, thickness=-1, color=(0, 95, 191))
                        img = proc.draw_point(img, pred, radius=3, thickness=-1, color=(255, 0, 0))
                    proc.image_save(img, "{}/draw/test/{}/mds1_ema/{}.{}".format(args.basePath, epo+1, imgID, args.imgType))
                    # endregion

                    # region 4.2 draw pictures with the predictions of mds2_ema
                    predArray_mds2_ema = preds_mds2_ema[iIdx].cpu().data.numpy().tolist()
                    for (pred_256, gt_256) in zip(predArray_mds2_ema, gtArray):
                        gt = [gt_256[0]*w/args.inpRes, gt_256[1]*h/args.inpRes]
                        pred = [pred_256[0]*w/args.inpRes, pred_256[1]*h/args.inpRes]
                        img = proc.draw_point(img, gt, radius=3, thickness=-1, color=(0, 95, 191))
                        img = proc.draw_point(img, pred, radius=3, thickness=-1, color=(255, 0, 0))
                    proc.image_save(img, "{}/draw/test/{}/mds2_ema/{}.{}".format(args.basePath, epo+1, imgID, args.imgType))
                    # endregion

                    # region 4.3 draw pictures with the predictions of mean
                    predArray_mean = preds_mean[iIdx].cpu().data.numpy().tolist()
                    for (pred_256, gt_256) in zip(predArray_mean, gtArray):
                        gt = [gt_256[0]*w/args.inpRes, gt_256[1]*h/args.inpRes]
                        pred = [pred_256[0]*w/args.inpRes, pred_256[1]*h/args.inpRes]
                        img = proc.draw_point(img, gt, radius=3, thickness=-1, color=(0, 95, 191))
                        img = proc.draw_point(img, pred, radius=3, thickness=-1, color=(255, 0, 0))
                    proc.image_save(img, "{}/draw/test/{}/mean/{}.{}".format(args.basePath, epo+1, imgID, args.imgType))
                    # endregion
            # endregion

            # region 5. clearing the GPU Cache
            del imgMap, outs_mds1_ema, outs_mds2_ema, preds_mds1_ema, preds_mds2_ema, preds_mean, errs_mds1_ema, accs_mds1_ema, errs_mds2_ema, accs_mds2_ema, errs_mean, accs_mean, _
            # endregion
    return predsArray_mds1_ema, accs_mds1_ema_counters.avg(), errs_mds1_ema_counters.avg(), predsArray_mds2_ema, accs_mds2_ema_counters.avg(), errs_mds2_ema_counters.avg(), predsArray_mean, accs_mean_counters.avg(), errs_mean_counters.avg()


def calSampleWeight_train(islabeledArray, args):
    # calculate the weight of the gt-sample (w = 1.0) and pseudo-label (w = args.cur_pw)
    islabeledArray = [islabeled.to(args.device, non_blocking=True) for islabeled in islabeledArray]  # 2 * [bs]
    sampleWeightArray = [islabeled.detach().float() for islabeled in islabeledArray]  # 2 * [bs]
    sampleWeightArray_pseudo = [args.pseudoWeight * torch.ones_like(sampleWeight) for sampleWeight in sampleWeightArray]  # 2 * [bs]
    sampleWeightArray = [setVariable(torch.where(islabeledArray[idx] > 0, sampleWeightArray[idx], sampleWeightArray_pseudo[idx]), args.device).unsqueeze(-1) for idx in range(len(islabeledArray))]
    return sampleWeightArray


def setVariable(dataItem, deviceID):
    return torch.autograd.Variable(dataItem.to(deviceID, non_blocking=True), requires_grad=True)


def setContent(dataArray, fmt):
    strContent = ""
    for dataIdx, dataItem in enumerate(dataArray):
        if dataIdx == len(dataArray)-1:
            strContent += "{}".format(format(dataItem, fmt))
        else:
            strContent += "{}, ".format(format(dataItem, fmt))
    return strContent


def setArgs(args, params):
    dict_args = vars(args)
    if params is not None:
        for key in params.keys():
            if key in dict_args.keys():
                dict_args[key] = params[key]
    for key in dict_args.keys():
        if dict_args[key] == "True": dict_args[key] = True
        if dict_args[key] == "False": dict_args[key] = False
    return argparse.Namespace(**dict_args)


def exec(expMark="MDSs", params=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = initArgs(params)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.experiment = "{}_{}({}_{})_{}".format(expMark, args.dataSource, args.trainCount, args.labelRatio, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    args.basePath = "{}/{}".format(glob.expr, args.experiment)
    glob.setValue("logger", Logger(args.experiment, consoleLevel="L1"))
    main(args)


def initArgs(params=None):
    # region 1. Parameters
    parser = argparse.ArgumentParser(description="Pose Estimation with SSL")
    # Dataset setting
    parser.add_argument("--dataSource", default="Mouse", choices=["Mouse", "Pranav", "Fly", "AP10K", "LSP", "FLIC"])
    parser.add_argument("--trainCount", default=100, type=int)
    parser.add_argument("--validCount", default=500, type=int)
    parser.add_argument("--labelRatio", default=0.3, type=float)
    # Model structure
    parser.add_argument("--mCount", default=5, type=int)
    parser.add_argument("--nStack", default=3, type=int, help="the number of stage in Multiple Pose Model")
    parser.add_argument("--inpRes", default=256, type=int, help="model input resolution (default: 256)")
    parser.add_argument("--outRes", default=64, type=int, help="model output resolution (default: 64)")
    # Training strategy
    parser.add_argument("--epochs", default=600, type=int, help="the number of total epochs")
    parser.add_argument("--trainBS", default=2, type=int, help="the batchSize of training")
    parser.add_argument("--inferBS", default=8, type=int, help="the batchSize of infering")
    parser.add_argument("--lr", default=2.5e-4, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=0, type=float, help="weight decay (default: 0)")
    parser.add_argument("--power", default=0.9, type=float, help="power for learning rate decay")
    # Data augment
    parser.add_argument("--useFlip", default="True", help="whether add flip augment")
    parser.add_argument("--scaleRange", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder", default=8, type=int, help="number of occluder to add in")
    # Data augment (to teacher in Mean-Teacher)
    parser.add_argument("--scaleRange_ema", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange_ema", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion_ema", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder_ema", default=8, type=int, help="number of occluder to add in")
    # Pseudo-Label
    parser.add_argument("--pseudoWeight_max", default=0.7, type=float, help="the max PCT of pseudo-labels to select")
    parser.add_argument("--pseudoWeight_rampup", default=100, type=int, help="length of the pseudo-labels' PCT ramp-up")
    parser.add_argument("--uncPCT_max", default=0.7, type=float, help="the max PCT of pseudo-labels to select")
    parser.add_argument("--uncPCT_rampup", default=50, type=int, help="length of the pseudo-labels' PCT ramp-up")
    parser.add_argument("--augSampleNum", default=5, type=int, help="the forward time of MC-Dropout")
    parser.add_argument("--distThrMax", default=3.0, type=float)
    parser.add_argument("--scoreThr_min", default=0.5, type=float)
    # Hyper-parameter
    parser.add_argument("--poseWeight", default=10.0, type=float, help="the weight of pose loss (default: 10.0)")
    parser.add_argument("--consWeight_max", default=20.0, type=float, help="the max weight of consistency loss")
    parser.add_argument("--consWeight_rampup", default=50, type=int, help="length of the consistency loss ramp-up")
    parser.add_argument("--PALWeight_max", default=0.005, type=float, help="the max weight of PAL")
    parser.add_argument("--PALWeight_rampup", default=500, type=int, help="the length of the PAL' weight ramp-up")
    # mean-teacher
    parser.add_argument("--ema_decay", default=0.999, type=float, help="ema variable decay rate (default: 0.999)")
    # misc
    parser.add_argument("--pck_thr", default=0.2, type=float)
    parser.add_argument("--program", default="SSL-Pose_v7.6.20220926.2")
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()
