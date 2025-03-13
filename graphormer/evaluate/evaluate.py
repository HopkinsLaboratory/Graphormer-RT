import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math
import sys
from os import path
import pickle
from tqdm import tqdm
import csv
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.append( path.dirname(   path.dirname( path.abspath(__file__) ) ) )
from pretrain import load_pretrained_model

import logging
from sklearn.linear_model import LinearRegression

def import_data(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        data=[]
        for row in r:
            data.append(row)
        return data


def gen_histogram(d_set):
    n, bins, patches = plt.hist(x=d_set,color='darkmagenta',
                            alpha=0.7, rwidth=0.85, bins=100)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Absolute Error with Retention Time / min')
    plt.ylabel('Frequency')
    m = np.mean(d_set)
    std = np.std(d_set)
    title = r"$\mu$ = " + str(np.round(m, 4)) + r"   $\sigma$ = " + str(np.round(std, 4) )
    plt.title(title)
    # plt.show()


def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args) 
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    # initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]

    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    name = str(checkpoint_path).split('/')[-1].split('.')[0]
    del model_state
    model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    task.load_dataset(split)

    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    smilesL = []
    methodL = []
    stdL = []

    with torch.no_grad():
        model.eval()

        for i, sample in enumerate(progress): ## Grabbing batched input, SMILES
            sample = utils.move_to_cuda(sample)

            y,std = model(**sample["net_input"])
            info = np.asarray(sample["net_input"]['batched_data']['smiles'])
            sm = info[:, 0]
            method = info[:, 1]
            smilesL.extend(sm)
            methodL.extend(method)
            y = y[:, :].reshape(-1)
            std = std[:, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            std = std.detach().cpu()
            std = [s.item() for s in std]

            stdL.extend(std)
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

        y_pred = np.asarray(y_pred, dtype=np.float64) * 1000
        y_true = np.asarray(y_true, dtype=np.float64) * 1000
        std = (np.asarray(stdL, dtype=np.float64)) * 1000 
        methodL = np.asarray(methodL)
        smilesL = np.asarray(smilesL)

    if use_pretrained:
        if cfg.task.pretrained_model_name == "pcqm4mv1_graphormer_base":
            evaluator = ogb.lsc.PCQM4MEvaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv1Evaluator: {result_dict}')
        elif cfg.task.pretrained_model_name == "pcqm4mv2_graphormer_base":
            evaluator = ogb.lsc.PCQM4Mv2Evaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv2Evaluator: {result_dict}')
    else: 
        if args.metric == "auc":
            auc = roc_auc_score(y_true, y_pred)
            logger.info(f"auc: {auc}")
        elif args.metric == "mae":
            mae = np.mean(np.abs(y_true - y_pred))
            logger.info(f"mae: {mae}")
        else: 
            seconds = True
            if seconds:
                y_pred *= 60
                y_true *= 60
            ae = np.abs(y_true - y_pred)
            mae = np.mean(ae) 
            rmse = math.sqrt(np.mean((y_true - y_pred) ** 2)) 
            mse = np.mean((y_true - y_pred) ** 2) 
            error = (y_true - y_pred)
            m_error = np.mean(error) 
            e_rel = 100*(np.abs(y_true - y_pred) / y_true)
            # for i in range(len(e_rel)):
            #     print(round(y_true[i], 1), round(y_pred[i], 1), round(ae[i], 1), round(std[i], 1), smilesL[i])


            indices = np.where(y_true > 0.66 * 60)
            error_onemin = ae[indices]

            save = args.save_path

            if save != 'None':
                if not save.endswith('.csv'):
                    raise ValueError("The save path must be a valid .csv file")
                stack = np.column_stack((smilesL, methodL, y_true, y_pred, std))
                with open(save, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["SMILES", "Method", 'True RT', "Predicted RT", 'STD'])
                    for row in stack:
                        writer.writerow(row)
                print("SAVED PREDICTIONS")

            logger.info(f"mae: {mae:.2f}")
            logger.info(f"rmse: {rmse:.2f}")
            logger.info(f"error: {m_error:.2f}")
            
def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )

    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            print("hi")
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)

if __name__ == '__main__':
    main()