# -*- coding: utf-8 -*-
import argparse
import datetime
import torch
from torch.optim.adam import Adam as TorchAdam
from torch.utils.data.dataloader import DataLoader as TorchDataLoader

import GLOB as glob
import datasources
import datasets
import models

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc
from utils.evaluation import EvaluationUtils as eval
from utils.losses import GateJointMSELoss, AvgCounter, AvgCounters

import matplotlib
matplotlib.use('Agg')


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.global_step = 0
    args.mds1_best_acc = -1.
    args.mds1_best_epoch = 0

    # region 1. dataloader initialize
    loadingTM = datetime.datetime.now()
    # data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgType, args.pck_ref = dataSource.kpsCount, dataSource.imgType, dataSource.pck_ref
    # train-set dataloader
    trainDS = datasets.__dict__["DS"]("train", labeledData, means, stds, isAug=True, isDraw=False, **vars(args))
    trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=False, pin_memory=True, drop_last=False)
    # valid-set dataloader
    validDS = datasets.__dict__["DS"]("valid", validData, means, stds, isAug=False, isDraw=False, **vars(args))
    validLoader = TorchDataLoader(validDS, batch_size=args.inferBS, shuffle=False, pin_memory=True, drop_last=False)
    logger.print("L1", "=> initialized {} Dataset loaders".format(args.dataSource), start=loadingTM)
    # endregion

    # region 2. modules initialize
    loadingTM = datetime.datetime.now()
    # Mean-Teacher Module 1
    mds1_model = models.__dict__["HG"](args.kpsCount, args.nStack).to(args.device)
    mds1_optim = TorchAdam(mds1_model.parameters(), lr=args.lr, weight_decay=args.wd)
    hg_pNum = sum(p.numel() for p in mds1_model.parameters())
    logc = "=> initialized HG models (nStack: {}, params: {})".format(args.nStack, format(hg_pNum / 1000000.0, ".2f"))
    logger.print("L1", logc, start=loadingTM)
    # endregion

    # region 3. iteration
    logger.print("L1", "=> training start")
    log_dataArray = []
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()

        # region 3.3 model training and validating
        mds1_pec_loss = train(trainLoader, mds1_model, mds1_optim, args)
        preds_mds1_ema, mds1_ema_accs, mds1_ema_errs = validate(validLoader, mds1_model, args, epo)
        # endregion

        # region 3.4 model selection & storage
        # region 3.4.1 mds1
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
            'state_dict_ema': mds1_model.state_dict(),
            'optim': mds1_optim.state_dict()
        }, mds1_is_best, ckptPath="{}/ckpts/mds1".format(args.basePath))
        # endregion
        # endregion

        # region 3.6 log storage
        log_dataArray.append({"mds1_pec_loss": mds1_pec_loss, "mds1_ema_errs": mds1_ema_errs, "mds1_ema_accs": mds1_ema_accs})
        if epo % 5 == 0:  # epo > 0 and epo % 5 == 0
            comm.json_save(log_dataArray, "{}/logs/logData.json".format(args.basePath), isCover=True)
        if epo == 0:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)
        # endregion

        # region 3.7 output result
        fmtc = "[{}/{}] pec_loss: {} | err: {}, acc: {} | best acc: {} (epo: {}) | kps_err:[{}], kps_acc:[{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"),
                           format(mds1_pec_loss, ".5f"),
                           format(mds1_ema_errs[-1], ".2f"), format(mds1_ema_accs[-1], ".3f"), format(args.mds1_best_acc, ".3f"), format(args.mds1_best_epoch, "3d"),
                           setContent(mds1_ema_errs[0:len(mds1_ema_errs)-1], ".2f"), setContent(mds1_ema_accs[0:len(mds1_ema_accs)-1], ".3f"))
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, All executing finished...]".format(args.experiment), start=allTM)


def train(trainLoader, mds1_model, mds1_optim, args):
    mds1_pec_counter = AvgCounter()
    pose_lossFunc = GateJointMSELoss(nStack=args.nStack, useKPsGate=True).to(args.device)
    mds1_model.train()
    for bat, (imgMap, kpsHeatmap, meta) in enumerate(trainLoader):
        mds1_optim.zero_grad()
        # region 1. data organize
        imgMap = torch.autograd.Variable(imgMap.to(args.device, non_blocking=True), requires_grad=True)
        kpsHeatmap = torch.autograd.Variable(kpsHeatmap.to(args.device, non_blocking=True), requires_grad=True)
        kpsGate = torch.autograd.Variable(meta['kpsWeight'].to(args.device, non_blocking=True), requires_grad=True)
        # endregion

        # region 2. model forward
        outs_mds1 = mds1_model(imgMap)
        # endregion

        # region 5. pose estimation constraint
        mds1_pec_loss = 0.
        # region 5.1 calculate mds1_pose_loss (using teacher's preds to cal the pose_loss)
        mds1_pec_sum, mds1_pec_count = 0., 0
        loss, n = pose_lossFunc(outs_mds1, kpsHeatmap, kpsGate)
        mds1_pec_sum += loss
        mds1_pec_count += n
        mds1_pec_loss += args.poseWeight * ((mds1_pec_sum / mds1_pec_count) if mds1_pec_count > 0 else mds1_pec_sum)
        mds1_pec_counter.update(mds1_pec_loss.item(), mds1_pec_count)
        # endregion
        # endregion

        # region 7. calculate total loss & update model
        # cal total loss
        mds1_total_loss = mds1_pec_loss
        # backward
        mds1_total_loss.backward()  # retain_graph=True
        mds1_optim.step()

        # region 8. clearing the GPU Cache
        del kpsHeatmap, kpsGate, imgMap
        del outs_mds1
        del mds1_pec_loss
        del mds1_total_loss
        # endregion
    return mds1_pec_counter.avg


def validate(validLoader, mds1_model_ema, args, epo):
    errs_mds1_ema_counters, accs_mds1_ema_counters = AvgCounters(), AvgCounters()
    mds1_model_ema.eval()
    predsArray_mds1_ema = []
    with torch.no_grad():
        for bat, (imgMap, kpsHeatmap, meta) in enumerate(validLoader):
            # region 1. data organize
            imgMap = imgMap.to(args.device, non_blocking=True)
            bs, k, _, _ = kpsHeatmap.shape
            # endregion

            # region 2. model forward
            outs_mds1_ema = mds1_model_ema(imgMap)[:, -1].cpu()  # 多模型预测结果 [bs, nstack, k, h, w]
            preds_mds1_ema, _ = proc.kps_fromHeatmap(outs_mds1_ema, meta['center'], meta['scale'], [args.outRes, args.outRes])  # kps_fromHeatmap(heatmap, cenMap, scale, res)

            predsArray_mds1_ema += preds_mds1_ema
            # endregion

            # region 3. calculate the error and accuracy
            errs_mds1_ema, accs_mds1_ema = eval.acc_pck(preds_mds1_ema, meta['kpsMap'], args.pck_ref, args.pck_thr)
            for idx in range(k+1):
                errs_mds1_ema_counters.update(idx, errs_mds1_ema[idx].item(), bs if idx < k else bs*k)  # bs*k 有问题，待调查。
                accs_mds1_ema_counters.update(idx, accs_mds1_ema[idx].item(), bs if idx < k else bs*k)
            # endregion

            # region 4. output the predictions
            if epo % 5 == 0:
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
            # endregion

            # region 5. clearing the GPU Cache
            del imgMap, outs_mds1_ema, preds_mds1_ema, errs_mds1_ema, accs_mds1_ema, _
            # endregion
    return predsArray_mds1_ema, accs_mds1_ema_counters.avg(), errs_mds1_ema_counters.avg()


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


def exec(expMark="HG", params=None):
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
    parser.add_argument("--dpoForwardNum", default=5, type=int, help="the forward time of MC-Dropout")
    parser.add_argument("--dpoSelNum", default=3, type=int, help="the forward time of MC-Dropout")
    # Training strategy
    parser.add_argument("--epochs", default=999, type=int, help="the number of total epochs")
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
    # Hyper-parameter
    parser.add_argument("--poseWeight", default=10.0, type=float, help="the weight of pose loss (default: 10.0)")
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
