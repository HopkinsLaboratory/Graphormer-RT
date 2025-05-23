from re import T
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import math
from tqdm import trange
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Mapping, Sequence, Tuple
from numpy import mod
import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq.dataclass.configs import FairseqDataclass
import matplotlib.pyplot as plt
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("rmse_HILIC", dataclass=FairseqDataclass)
class SID(FairseqCriterion):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def __init__(self, task):
        super().__init__(task)
        self.threshold = 1e-8 
        self.eps = 1e-8
        self.torch_device = 'cuda'

    def forward(self, model, sample, update_num=10, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        sample_size = sample["nsamples"]
        #print(sample_size)
        values = model(**sample["net_input"])
        label = sample['target'] 
        #loss = self.sid(values, label)
        mu =  values

        mu = mu.squeeze()
        # std = std.squeeze()
        label=label.squeeze()

        loss = self.rmse(mu, label)
        
        logging_output = {
            "loss": loss,
            "sample_size": 1,
            "ntokens": 1,
            "nsentences": 1,
            "ncorrect": 0,
        
        } ## Arbitary parameters you want to spit out for training

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def log_loss(self, mu, std, target):
        std = std.float() + 5e-2 ## adding a residual to help with numerical stability in early epochs. 
        mu = mu.float()
        target = target.float()
        loss = ((mu - target)**2 / (2 * std**2) +std) * 1000 ## s13321-019-0374-3 

        loss = torch.mean(loss)
        return loss

    def rmse(self, model, target):
        loss = torch.sqrt(torch.mean((model-target)**2))

        return loss * 1000

