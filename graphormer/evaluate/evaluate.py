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
    with torch.no_grad():
        model.eval()

        for i, sample in enumerate(progress): ## Grabbing batched input, SMILES
            sample = utils.move_to_cuda(sample)

            y = model(**sample["net_input"])
            # print(sample["net_input"]['batched_data']['smiles'])
            info = np.asarray(sample["net_input"]['batched_data']['smiles'])
            sm = info[:, 0]
            method = info[:, 1]
            smilesL.extend(sm)
            methodL.extend(method)
            y = y[:, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

        # print(y_pred, "PRED")
        # print(y_true, "TRUE")
        y_pred = np.asarray(y_pred, dtype=np.float64) * 1000
        y_true = np.asarray(y_true, dtype=np.float64) * 1000
        methodL = np.asarray(methodL)
        smilesL = np.asarray(smilesL)

        save = False
        if save:
            stack = np.column_stack((smilesL, methodL, y_true, y_pred, np.abs(y_true - y_pred)))
            with open('/home/cmkstien/Graphormer_RT/results/RT_Predictions.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["SMILES", "Method", "True RT", "Predicted RT", "Absolute Error"])
                for row in stack:
                    writer.writerow(row)
            print("SAVED PREDICTIONS")
        
        # print(y_pred)
        # print(y_true)
        # exit()
        ## UNCOMMENT TO RESTORE 
        # y_ext = y_pred[-215:]
        # y_true_ext = y_true[-215:]
        # methodL_ext = methodL[-215:]
        # smilesL_ext = smilesL[-215:]

        # y_pred = y_pred[:-215]
        # y_true = y_true[:-215]
        # # print(y_pred)
        # # print(y_true)
        # # exit()
        # methodL = methodL[:-215]
        # smilesL = smilesL[:-215]


    # save predictions
    # evaluate pretrained models
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
            seconds = False
            if seconds:
                y_pred *= 60
                y_true *= 60
            ae = np.abs(y_true - y_pred)
            mae = np.mean(ae) * 60 
            rmse = math.sqrt(np.mean((y_true - y_pred) ** 2)) 
            mse = np.mean((y_true - y_pred) ** 2) 
            error = (y_true - y_pred)
            m_error = np.mean(error) 

            # print(y_true_ext[:10])
            # print(y_ext[:10])

            ## RESTORE
            # error_ext = np.abs(y_true_ext - y_ext)
            error_29 = []
            error_275 = []
            error_127 = []

            # for i in range(len(error_ext)):
            #     if methodL_ext[i] == '0029':
            #         error_29.append(error_ext[i])
            #         # print(methodL_ext[i], error_ext[i])
            #     elif methodL_ext[i] == '0275':
            #         error_275.append(error_ext[i])
            #     elif methodL_ext[i] == '0127':
            #         error_127.append(error_ext[i])

            # for i in range(len(error)):
            #     if methodL[i] == '0029':
            #         error_29.append(ae[i])
            #     elif methodL[i] == '0275':
            #         error_275.append(ae[i])
            #     elif methodL[i] == '0127':
            #         error_127.append(ae[i])

            # m_error_29 = np.mean(error_29) * 60
            # m_error_275 = np.mean(error_275) * 60
            # m_error_127 = np.mean(error_127) * 60

            # mae_ext = np.mean(np.abs(y_true_ext - y_ext)) * 60
            # rmse_ext = math.sqrt(np.mean((y_true_ext - y_ext) ** 2)) * 60
            # mse_ext = np.mean((y_true_ext - y_ext) ** 2) * 60
            # print("MAE 29: ", m_error_29)
            # print("MAE 275: ", m_error_275)
            # print("MAE 127: ", m_error_127)
            # print(len(error_29), len(error_275), len(error_127))

            # print("RMSE EXTERNAL: ", rmse_ext)
            # print("MSE EXTERNAL: ", mse_ext)

            logger.info(f"mae: {mae}")
            logger.info(f"rmse: {rmse}")
            logger.info(f"mse: {mse}")
            logger.info(f"error: {m_error}")

            ### Plotting ML Correlation Diagram
            R2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
            max_val = np.max([y_true, y_pred]) + 2
            text = '$R^2$ = ' + str(np.round(R2, 2)) + '\nMAE = ' + str(np.round(mae, 2))+ ' sec' +  '\nRMSE = ' + str(np.round(rmse, 2)) + ' sec'
            box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.24, 0.8, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=box, horizontalalignment='center')
            plt.scatter(y_true, y_pred, alpha=0.6, color='black', s=0.5)
            plt.plot([0, max_val], [0, max_val], color='red',linestyle = 'dashed')
            plt.xlabel('True RT / min')
            plt.ylabel('Predicted RT / min')
            plt.title('RT Prediction, Average Error: ' + str(np.round(mae, 2)) + ' seconds (MAE)')
            plt.savefig('/home/cmkstien/Graphormer_RT/HuanLab/ML_Correlation_Diagram.png', dpi=300)
            plt.clf()
            
            ### Plotting Error Histogram
            gen_histogram(error)
            plt.savefig('/home/cmkstien/Graphormer_RT/HuanLab/Error_Histogram.png', dpi=300)
            plt.clf()
            ###Plotting Error Against RT
            plt.scatter(y_true, ae, alpha=0.5, s=5, color='blue')
            plt.xlabel('True RT / min')
            plt.ylabel('MAE / min')
            plt.title('MAE vs RT')
            plt.savefig('/home/cmkstien/Graphormer_RT/HuanLab/Error_vs_RT.png', dpi=300)
            plt.clf()
            ###Plotting Error Against Method
            plt.figure(figsize=(40, 20))
            arr = np.column_stack((methodL, ae))
            df = pd.DataFrame(arr, columns=['Method', 'MAE'])
            df['Method'] = df['Method'].astype(str)
            df['MAE'] = df['MAE'].astype(float)
            df = df.groupby('Method').mean()
            df = df.reset_index()
            df = df.sort_values('MAE')
            plt.bar(df['Method'], df['MAE'])
            print(df['Method'].shape, df['MAE'].shape)
            plt.xticks(rotation=90)
            plt.xlabel('Method')
            plt.ylabel('MAE / min')
            plt.title('MAE vs Method')
            plt.savefig('/home/cmkstien/Graphormer_RT/HuanLab/Error_vs_Method.png', dpi=300)
            plt.clf()

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