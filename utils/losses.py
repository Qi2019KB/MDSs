# -*- coding: utf-8 -*-
import torch
from torch import nn
from .process import ProcessUtils as proc


class GateJointMSELoss(nn.Module):
    def __init__(self, nStack=1, useKPsGate=False, useSampleWeight=False):
        super(GateJointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.useKPsGate = useKPsGate
        self.useSampleWeight = useSampleWeight

    def forward(self, preds, gts, kpsGate=None, sampleWeight=None):
        bs, k = preds.size(0), preds.size(1) if self.nStack == 1 else preds.size(2)
        kpsGate_clone = torch.ones([bs, k]) if kpsGate is None else kpsGate.detach()
        kpsNum = proc.kps_getLabeledCount(kpsGate_clone)
        combined_loss = []
        for nIdx in range(self.nStack):
            v1 = preds.reshape((bs, k, -1)) if self.nStack == 1 else preds[:, nIdx].reshape((bs, k, -1))
            v2 = gts.reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if self.useKPsGate: loss = loss.mul(kpsGate_clone)
            if self.useSampleWeight and sampleWeight is not None: loss = loss.mul(sampleWeight)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), self.nStack*kpsNum


class GateJointDistLoss(nn.Module):
    def __init__(self, nStack=1, useKPsGate=False, useSampleWeight=False):
        super(GateJointDistLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.nStack = nStack
        self.useKPsGate = useKPsGate
        self.useSampleWeight = useSampleWeight

    def forward(self, preds1, preds2, kpsGate=None, sampleWeight=None):
        bs, k = preds1.size(0), preds1.size(1) if self.nStack == 1 else preds1.size(2)
        kpsGate_clone = torch.ones([bs, k]) if kpsGate is None else kpsGate.detach()
        kpsNum = proc.kps_getLabeledCount(kpsGate_clone)
        combined_loss = []
        for nIdx in range(self.nStack):
            v1 = preds1.reshape((bs, k, -1)) if self.nStack == 1 else preds1[:, nIdx].reshape((bs, k, -1))
            v2 = preds2.reshape((bs, k, -1)) if self.nStack == 1 else preds2[:, nIdx].reshape((bs, k, -1))
            loss = self.criterion(v1, v2).mean(-1)
            if self.useKPsGate: loss = loss.mul(kpsGate_clone)
            if self.useSampleWeight and sampleWeight is not None: loss = loss.mul(sampleWeight)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss.sum(), self.nStack*kpsNum


class AvgCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0. if self.count == 0 else self.sum / self.count


class AvgCounters(object):
    def __init__(self, num=1):
        self.counters = [AvgCounter() for i in range(num)]
        self.reset()

    def reset(self):
        for counter in self.counters:
            counter.reset()

    def update(self, idx, val, n=1):
        self.check_idx(idx)
        self.counters[idx].update(val, n)

    def avg(self):
        return [item.avg for item in self.counters]

    def sum(self):
        return [item.sum for item in self.counters]

    def check_idx(self, idx):
        if len(self.counters) < idx + 1:
            for i in range(len(self.counters), idx + 1):
                self.counters.append(AvgCounter())


