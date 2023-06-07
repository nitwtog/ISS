import argparse
import logging
import os
import pickle
from pathlib import Path
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import csv
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BertForMaskedLM
)

from tracIn import get_grads_score_of_external, get_grads
from model.model import BertForMLMandFT

import model.utils as utils
from utils.collator import DataCollatorForLanguageModeling
from utils.dataset import preprocess
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# label num for each task
dapt_tasks = {
    "chemprot": 13,
    "citation_intent": 6,
    "hyp": 2,
    "imdb": 2,
    "rct-20k": 5,
    "sciie": 7,
    "ag": 4,
    "amazon": 2,
}

def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/'))
    parser.add_argument('--num_process', type=int, default=4)
    parser.add_argument('--mode', type=str, default='tracIn')
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--from_external', type=str,default=None,choices=['large_external.csv','small_external.csv',None])


    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--save_models', type=int, default=1)
    parser.add_argument('--mlm_weight', type=int, default=5)

    # hyper-parameters
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--test_bsz', type=int, default=1)
    parser.add_argument('--external_bsz', type=int, default=1)

    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--modeldir', default='temp', type=str)
    parser.add_argument('--cuda', default='0', type=str)

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def load_data(tokenizer, args):
    # load train data
    print("[Data Info] Loading train data", flush=True)
    data_files = {"train": f'https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/train.csv'}
    train_dataset = load_dataset('csv', data_files=data_files)
    train_dataset = preprocess(train_dataset, tokenizer, args.num_process)
    train_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)
    train_dataload = DataLoader(train_dataset, shuffle=False, collate_fn=train_data_collator,
                                batch_size=args.bsz)

    # load valid data
    print("[Data Info] Loading dev data", flush=True)
    data_files = {"train": f'https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/dev.csv'}
    valid_dataset = load_dataset('csv', data_files=data_files)
    valid_dataset = preprocess(valid_dataset, tokenizer, args.num_process)
    valid_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)
    valid_dataload = DataLoader(valid_dataset, shuffle=False, collate_fn=valid_data_collator,
                                batch_size=args.bsz)

    # load test data
    print("[Data Info] Loading test data", flush=True)
    data_files = {"train": f'https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/test.csv'}
    test_dataset = load_dataset('csv', data_files=data_files)
    test_datasets = preprocess(test_dataset, tokenizer, args.num_process)
    test_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)
    test_dataload = DataLoader(test_datasets, shuffle=False, collate_fn=test_data_collator, batch_size=args.bsz)

    return train_dataset, train_dataload, valid_dataload, test_dataload


def evaluate(model, data_loader, device):
    model.eval()
    avg_loss = utils.ExponentialMovingAverage()
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            id_list = batch.pop("id", None)
            labels = batch.pop("cls_labels", None)
            logits = model(**batch).logits
            _, preds = logits.max(dim=-1)

            if labels.shape[-1] == 1:
                label_list.append(int(labels.squeeze(-1).cpu()))
                pred_list.append(int(preds.cpu()))
                continue

            label_list.extend(labels.squeeze(-1).cpu().numpy().tolist())
            pred_list.extend(preds.cpu().numpy().tolist())
        f1 = f1_score(y_true=label_list, y_pred=pred_list, average='macro')

    return f1, avg_loss.get_metric()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_seed(args.seed)

    output_dir = Path(os.path.join(args.ckpt_dir, args.modeldir.format()))
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    # pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = dapt_tasks[args.task_name]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_num_labelslower_case=args.do_lower_case)
    model = BertForMLMandFT.from_pretrained(args.model_name, config=config)
    model.init_weights()
    model.set_args(args)
    model.to(device)

    if args.mode == "train":
        train_dataset, train_loader, dev_loader, test_loader = load_data(tokenizer, args)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon,
            correct_bias=args.bias_correction
        )

        # Use suggested learning rate scheduler
        num_training_steps = len(train_dataset) * args.epochs // args.bsz
        warmup_steps = num_training_steps * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

        best_dev_epoch, best_dev_accuracy, test_accuracy = 0, 0, 0
        for epoch in range(args.epochs):
            model.train()
            avg_loss = utils.ExponentialMovingAverage()
            pbar = tqdm(train_loader)

            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                id_list = batch.pop("id", None)
                # labels = batch.pop("cls_labels", None)
                # labels = batch.pop("labels", None)
                model.zero_grad()

                # batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
                outpus = model(**batch)

                # (1) backward
                total_loss = 0.0
                loss = outpus.loss

                total_loss += loss.item()
                loss.backward()

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                avg_loss.update(total_loss)
                pbar.set_description(f'epoch: {epoch: d}, '
                                     f'loss: {avg_loss.get_metric(): 0.4f}, '
                                     f'lr: {optimizer.param_groups[0]["lr"]: .3e}')

            if args.save_models:
                s = Path(str(output_dir) + '/epoch' + str(epoch))
                if not s.exists():
                    s.mkdir(parents=True)
                model.save_pretrained(s)
                tokenizer.save_pretrained(s)
                torch.save(args, os.path.join(s, "training_args.bin"))

            # test after one epoch
            dev_accuracy, dev_loss = evaluate(model, dev_loader, device)
            logger.info(f'Epoch: {epoch}, '
                        f'Loss: {avg_loss.get_metric(): 0.4f}, '
                        f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                        f'Dev_Accuracy: {dev_accuracy}')

            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_dev_epoch = epoch
                test_accuracy, test_loss = evaluate(model, test_loader, device)
                logger.info(f'**** Test Accuracy: {test_accuracy}, Test_Loss: {test_loss}')
                if args.save_models:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))

        logger.info(f'**** Best dev metric: {best_dev_accuracy} in Epoch: {best_dev_epoch}')
        logger.info(f'**** Best Test metric: {test_accuracy} in Epoch: {best_dev_epoch}')

        last_test_accuracy, last_test_loss = evaluate(model, test_loader, device)
        logger.info(f'Last epoch test_accuracy: {last_test_accuracy}, test_loss: {last_test_loss}')
    if args.mode == "tracIn":
        # load model parameters
        model = model.from_pretrained(r'/workspace/chemprot/classify-baseline/saved_models/chemprot')
        model.set_args(args)
        model.to(device)

        for name, param in model.named_parameters():
            param.requires_grad = True
        for name, param in model.bert.named_parameters():
            if '11' in name:
                continue
            if 'encoder' in name or 'embedding' in name:
                param.requires_grad = False
        model.cls.predictions.decoder.weight.requires_grad = True

        # load external data
        print("[Data Info] Loading external data", flush=True)
        data_files = {"train": f'https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/{args.from_external}'}
        print(data_files)
        external_datasets = load_dataset('csv', data_files=data_files)
        external_datasets = preprocess(external_datasets, tokenizer, args.num_process)
        external_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)
        external_dataload = DataLoader(external_datasets, shuffle=False, collate_fn=external_data_collator,
                                       batch_size=args.external_bsz)

        # load train data
        print("[Data Info] Loading test data", flush=True)
        data_files = {"train": f'https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/train.csv'}
        print(data_files)
        test_datasets = load_dataset('csv', data_files=data_files)
        test_datasets = preprocess(test_datasets, tokenizer, args.num_process)
        test_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)
        test_dataload = DataLoader(test_datasets, shuffle=False, collate_fn=test_data_collator,
                                   batch_size=args.test_bsz)

        test_accuracy, test_loss = evaluate(model, test_dataload, device)
        print(f'test f1 socre : {test_accuracy}')
        model.train()

        grads_test_weight, grads_test_bias, index_list_test = get_grads(args, model, test_dataload, device)

        score_list, index_list_ex = get_grads_score_of_external(args, model, external_dataload, grads_test_weight,
                                                                device)
        with open('external_score_ex' + str(args.external_bsz) + '_test' + str(args.test_bsz) + '.pickle', 'wb') as f:
            data = {'score_list': score_list, 'index': index_list_ex}
            pickle.dump(data, f)


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
