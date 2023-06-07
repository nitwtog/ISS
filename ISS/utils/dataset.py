from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
import numpy as np
import collections
import pandas as pd
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

FCInst = collections.namedtuple('FCInst', 'text label')
FCFeat = collections.namedtuple('FCFeat', 'input_ids attention_mask label_id')

def convert_instances_to_feature_tensors(instances: List[FCInst], tokenizer: PreTrainedTokenizer, max_seq_length: int) -> List[FCFeat]:
    features = []
    for idx, inst in enumerate(instances):
        batch_input = tokenizer.batch_encode_plus([inst.text], truncation=True, max_length=max_seq_length + 2)
        word_ids = batch_input['input_ids'][0]
        attention_mask = batch_input['attention_mask'][0]
        label_ids = inst.label
        features.append(FCFeat(input_ids=word_ids,
                               attention_mask=attention_mask,
                               label_id=label_ids,
                               ))
    return features


class FCDataset(Dataset):
    def __init__(self,
                 file: Union[str, None],
                 tokenizer: PreTrainedTokenizer,
                 pretrain_model_name: str,
                 number: int = -1,
                 max_text_len: int = 1000,
                 task: str = "cls",
                ) -> None:
        insts = []
        self.skip_num = 0
        self.tokenizer = tokenizer
        self.pretrain_model_name = pretrain_model_name
        self.task = task
        print(f"[Data Info] Reading file: {file}")

        data = pd.read_csv(file).values.tolist()

        if number > 0:
            data = data[:number]

        for sample in data:
            insts.append(FCInst(text=sample[0],
                                label=int(sample[1])))

        print(insts[1])

        self._features = convert_instances_to_feature_tensors(instances=insts,
                                                              tokenizer=tokenizer,
                                                              max_seq_length=max_text_len)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> FCFeat:
        return self._features[idx]

    def collate_fn(self, batch: List[FCFeat]):
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])

        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            batch[i] = FCFeat(
                input_ids=np.asarray(feature.input_ids + [self.tokenizer.pad_token_id] * padding_length),
                attention_mask=np.asarray(feature.attention_mask + [0] * padding_length),
                label_id=feature.label_id,
            )

        # print(batch)
        results = FCFeat(*(default_collate(samples) for samples in zip(*batch)))
        return results



def preprocess(dataset,tokenizer, num_process=1):
    padding = "max_length"
    def tokenize_function(examples):
        # Remove empty lines
        examples['text'] = [
            line for line in examples['text'] if line is not None and len(line) > 0 and not line.isspace()
        ]

        tokenized_examples = tokenizer(
            examples['text'],
            padding=padding,
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True,
        )
        if "label" in examples:
            tokenized_examples["cls_labels"] = examples["label"]
        for k in examples:
            if k not in tokenized_examples and k != 'text' and k != "rank" and k != "ids" and k != "id":
                tokenized_examples[k] = examples[k][:len(examples['text'])]
        return tokenized_examples
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_process,
        remove_columns=['text'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line_by_line",
    )
    return  tokenized_datasets['train']