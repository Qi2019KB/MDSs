# -*- coding: utf-8 -*-
import math
import copy
import torch
from utils.process import ProcessUtils as proc
from utils.evaluation import EvaluationUtils as eval


class BusinessUtils:
    def __init__(self):
        pass

    @classmethod
    def pseudo_cal_unc(cls, imageIDs, preds_gt, preds_mds1, scores_mds1, augPredsArray_mds1, augScoresArray_mds1, preds_mds2, scores_mds2, augPredsArray_mds2, augScoresArray_mds2, args):
        norms = eval.acc_pck_pseudo_norm(imageIDs, preds_gt, args.pck_ref)
        bsNum, kNum = augPredsArray_mds1.size(0), augPredsArray_mds1.size(1)
        pseudoArray_mds1, pseudoArray_mds2 = [], []
        for bsIdx in range(bsNum):
            for kIdx in range(kNum):
                # init mds1_kSample
                mds1_kSample = cls._initKSample(cls, bsIdx, kIdx, imageIDs[bsIdx], preds_gt, norms, preds_mds1, scores_mds1, augPredsArray_mds1, augScoresArray_mds1, args)
                # init mds2_kSample
                mds2_kSample = cls._initKSample(cls, bsIdx, kIdx, imageIDs[bsIdx], preds_gt, norms, preds_mds2, scores_mds2, augPredsArray_mds2, augScoresArray_mds2, args)
                pseudoArray_mds1.append(mds1_kSample)
                pseudoArray_mds2.append(mds2_kSample)
        for idx in range(len(pseudoArray_mds1)):
            pseudoArray_mds1[idx], pseudoArray_mds2[idx] = cls._calKSampleExterData(cls, pseudoArray_mds1[idx], pseudoArray_mds2[idx], args)
        return pseudoArray_mds1, pseudoArray_mds2

    @classmethod
    def pseudo_filter_mixUnc(self, pseudoArray, args):
        # scoreThr = self._calScoreThr(self, pseudoArray, args)
        # for pseudoItem in pseudoArray:
        #     pseudoItem["scoreOK"] = 0 if pseudoItem["score"] <= scoreThr else 1
        #     pseudoItem["unc"] = 999.0 if pseudoItem["score"] <= scoreThr else pseudoItem["unc"]
        uncThr = self._calUncValue(self, args.distThrMax*3)
        # pseudo-label filter
        selArray, selCounts, selErrs, selAccs = [], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)]
        for pseudoItem in pseudoArray:
            item = copy.deepcopy(pseudoItem)
            if item["unc"] <= uncThr:
                kID = int(item["kpID"].split("_")[-1])
                item["enable"] = 1
                selCounts[-1] += 1
                selCounts[kID] += 1
                selErrs[-1] += item["error"]
                selErrs[kID] += item["error"]
                selAccs[-1] += item["acc_flag"]
                selAccs[kID] += item["acc_flag"]
            else:
                item["enable"] = 0
            selArray.append(item)
        for idx in range(args.kpsCount+1):
            if selCounts[idx] > 0:
                selErrs[idx] = selErrs[idx] / selCounts[idx]
                selAccs[idx] = selAccs[idx] / selCounts[idx]
        # endregion
        return selArray, selCounts, selErrs, selAccs, uncThr

    @classmethod
    def pseudo_filter_mixUnc2(self, pseudoArray, args):
        scoreThr = self._calScoreThr(self, pseudoArray, args)
        for pseudoItem in pseudoArray:
            pseudoItem["scoreOK"] = 0 if pseudoItem["score"] < scoreThr else 1
            pseudoItem["unc"] = 999.0 if pseudoItem["score"] < scoreThr else pseudoItem["unc"]
        uncThr = self._calUncValue(self, args.distThrMax*3)
        # pseudo-label filter
        selArray, selCounts, selErrs, selAccs = [], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)], [0 for i in range(args.kpsCount+1)]
        for pseudoItem in pseudoArray:
            item = copy.deepcopy(pseudoItem)
            if item["unc"] <= uncThr:
                kID = int(item["kpID"].split("_")[-1])
                item["enable"] = 1
                selCounts[-1] += 1
                selCounts[kID] += 1
                selErrs[-1] += item["error"]
                selErrs[kID] += item["error"]
                selAccs[-1] += item["acc_flag"]
                selAccs[kID] += item["acc_flag"]
            else:
                item["enable"] = 0
            selArray.append(item)
        for idx in range(args.kpsCount+1):
            if selCounts[idx] > 0:
                selErrs[idx] = selErrs[idx] / selCounts[idx]
                selAccs[idx] = selAccs[idx] / selCounts[idx]
        # endregion
        return selArray, selCounts, selErrs, selAccs, scoreThr, uncThr

    @classmethod
    def preds_mean(cls, preds1, preds2):
        preds_mix = torch.stack([preds1, preds2], dim=-1)
        preds_mean = torch.mean(preds_mix, dim=-1)
        return preds_mean

    def _initKSample(self, bsIdx, kIdx, imageID, preds_gt, norms, preds_mds1, scores_mds1, augPredsArray_mds1, augScoresArray_mds1, args):
        k_id = "{}_{}".format(imageID, kIdx)
        k_coord, k_gt = preds_mds1[bsIdx][kIdx].cpu().data.numpy().tolist(), preds_gt[bsIdx][kIdx].cpu().data.numpy().tolist()
        k_error = proc.coord_distance(k_coord, k_gt)
        k_accFlag = eval.acc_pck_pseudo(k_error, [item["norm"] for item in norms if item["imageID"] == imageID][0], args.pck_thr)

        k_augCoords = augPredsArray_mds1[bsIdx][kIdx].cpu().data.numpy().tolist()  # bs,k,df,2
        k_augScores = [self._scoreFormat(self, item) for item in augScoresArray_mds1[0][kIdx].cpu().data.numpy().tolist()]
        k_scores = k_augScores + [self._scoreFormat(self, scores_mds1[0][kIdx].item())]  # bs,k,df
        k_score = k_scores[-1]  # self._calScoreFlag(self, k_scores) -- 各score连乘后的结果。

        k_augCoord = [sum([item[0] for item in k_augCoords]) / len(k_augCoords), sum([item[1] for item in k_augCoords]) / len(k_augCoords)]
        k_intDist = proc.coord_avgDistance(k_augCoords)

        k_sample = {"kpID": k_id, "coord": k_coord, "coord_gt": k_gt, "error": k_error, "acc_flag": k_accFlag, "coords_aug": k_augCoords, "coord_aug": k_augCoord,
                    "scores": k_scores, "score": k_score, "intDist": k_intDist}
        return k_sample

    def _calKSampleExterData(self, mds1_kSample, mds2_kSample, args):
        # cal extDist & aExtDist
        mds1_kSample["extDist"] = mds2_kSample["extDist"] = proc.coord_distance(mds1_kSample["coord"], mds2_kSample["coord"])
        mds1_kSample["aExtDist"] = mds2_kSample["aExtDist"] = proc.coord_distance(mds1_kSample["coord_aug"], mds2_kSample["coord_aug"])
        # cal intDist_lma & extDist_lma & aExtDist_lma
        mds1_kSample = self._lma_calKSampleLMAData(self, mds1_kSample, args.mds1_lma_cache)
        mds2_kSample = self._lma_calKSampleLMAData(self, mds2_kSample, args.mds2_lma_cache)

        mds1_kSample["mixDist"] = mds1_kSample["intDist_lma"] + ((mds1_kSample["extDist_lma"] + mds1_kSample["aExtDist_lma"])/2 if mds1_kSample["extDist_lma"] > 0 else mds1_kSample["aExtDist_lma"])
        mds2_kSample["mixDist"] = mds2_kSample["intDist_lma"] + ((mds2_kSample["extDist_lma"] + mds2_kSample["aExtDist_lma"])/2 if mds2_kSample["extDist_lma"] > 0 else mds2_kSample["aExtDist_lma"])

        mds1_kSample["intDistOK"] = 1 if mds1_kSample["intDist"] <= args.distThrMax else 0
        mds1_kSample["intDistOK_lma"] = 1 if mds1_kSample["intDist_lma"] <= args.distThrMax else 0
        mds1_kSample["extDistOK"] = 1 if mds1_kSample["extDist"] <= args.distThrMax else 0
        mds1_kSample["extDistOK_lma"] = 1 if mds1_kSample["extDist_lma"] <= args.distThrMax else 0
        mds1_kSample["aExtDistOK"] = 1 if mds1_kSample["aExtDist"] <= args.distThrMax else 0
        mds1_kSample["aExtDistOK_lma"] = 1 if mds1_kSample["aExtDist_lma"] <= args.distThrMax else 0
        mds2_kSample["intDistOK"] = 1 if mds2_kSample["intDist"] <= args.distThrMax else 0
        mds2_kSample["intDistOK_lma"] = 1 if mds2_kSample["intDist_lma"] <= args.distThrMax else 0
        mds2_kSample["extDistOK"] = 1 if mds2_kSample["extDist"] <= args.distThrMax else 0
        mds2_kSample["extDistOK_lma"] = 1 if mds2_kSample["extDist_lma"] <= args.distThrMax else 0
        mds2_kSample["aExtDistOK"] = 1 if mds2_kSample["aExtDist"] <= args.distThrMax else 0
        mds2_kSample["aExtDistOK_lma"] = 1 if mds2_kSample["aExtDist_lma"] <= args.distThrMax else 0

        mds1_kSample["unc"] = self._calUncValue(self, mds1_kSample["mixDist"]) if mds1_kSample["intDistOK_lma"] > 0 and mds1_kSample["extDistOK_lma"] > 0 and mds1_kSample["aExtDistOK_lma"] > 0 else 999.0
        mds2_kSample["unc"] = self._calUncValue(self, mds2_kSample["mixDist"]) if mds2_kSample["intDistOK_lma"] > 0 and mds2_kSample["extDistOK_lma"] > 0 and mds2_kSample["aExtDistOK_lma"] > 0 else 999.0
        return mds1_kSample, mds2_kSample

    def getLMAfromCache(self, lma_cache, kpID):
        targets = [item for item in lma_cache if item["kpID"] == kpID]
        if len(targets) == 0:
            item = {"kpID": kpID, "intDist": [], "extDist": [], "aExtDist": [], "intDist_lma": [], "extDist_lma": [], "aExtDist_lma": []}
            lma_cache.append(item)
            return item
        else:
            return targets[0]

    def _calScoreThr(self, dataArray, args):
        scores = [item["score"] for item in dataArray]
        # 方案一：
        scores_sorted = sorted(scores, reverse=True)
        scoreThr = scores_sorted[int((len(scores_sorted) - 1) * 0.5)]
        # 方案二：
        # scoreThr = max(args.scoreThr_min, sum(scores)/len(scores))
        return scoreThr

    def _scoreFormat(self, score):
        return max(0.0, min(1.0, score))

    def _calScoreFlag(self, scores):
        scoreFlag = 1.0
        for score in scores:
            scoreFlag *= score
        return scoreFlag

    def _calUncValue(self, mixDist):
        return 1.0 - math.exp(-mixDist/5)

    def _lma_calKSampleLMAData(self, mds1_kSample, lma_cache):
        lma_target = self.getLMAfromCache(self, lma_cache, mds1_kSample["kpID"])
        lma_target["intDist"].append(mds1_kSample["intDist"])
        lma_target["extDist"].append(mds1_kSample["extDist"])
        lma_target["aExtDist"].append(mds1_kSample["aExtDist"])

        intDist_lma = self._lma_variables(self, lma_target["intDist"])
        extDist_lma = self._lma_variables(self, lma_target["extDist"])
        aExtDist_lma = self._lma_variables(self, lma_target["aExtDist"])

        mds1_kSample["intDist_lma"] = intDist_lma
        mds1_kSample["extDist_lma"] = extDist_lma
        mds1_kSample["aExtDist_lma"] = aExtDist_lma

        lma_target["intDist_lma"].append(intDist_lma)
        lma_target["extDist_lma"].append(extDist_lma)
        lma_target["aExtDist_lma"].append(aExtDist_lma)
        return mds1_kSample

    def _lma_variables(self, sources):
        alphas = [0.5, 0.3, 0.2]
        if len(sources) == 0:
            return 999.0
        elif len(sources) == 1:
            return sources[-1]
        elif len(sources) == 2:
            return sources[-1] * (alphas[0] + alphas[1]) + sources[-2] * alphas[2]
        else:
            return sources[-1] * alphas[0] + sources[-2] * alphas[1] + sources[-3] * alphas[2]