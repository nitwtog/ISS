import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


def get_grads(args ,
              model: nn.Module,
              dataloader: DataLoader,
              device):
    dev = device
    # model.eval()
    model.to(dev)
    loss_weight_grads_list = []
    loss_bias_grads_list = []
    index_list = []
    for i, batch in tqdm(enumerate(dataloader), desc="--test", total=len(dataloader)):
        model.zero_grad()
        # with torch.no_grad():
        index = batch.pop("id", None)
        # cls_labels = batch.pop("cls_labels", None)
        labels = batch.pop("labels", None)
        for k,v in batch.items():
            batch[k] = v.to(dev)

        outputs = model(**batch)
        outputs.loss.backward()

        weight_grad = model.bert.encoder.layer[11].attention.output.dense.weight.grad.view(-1)
        # bias_grad = model.bert.encoder.layer[11].attention.output.dense.bias.grad.view(-1)
        loss_weight_grads_list.append(weight_grad.cpu().numpy().tolist())
        # loss_bias_grads_list.append(bias_grad.cpu().numpy().tolist())
        index_list.append(index.cpu().numpy().tolist())
        # if i == 2:
        #     model.zero_grad()
        #     break
    return loss_weight_grads_list, loss_bias_grads_list, index_list

def get_grads_score_of_external(args,
              model: nn.Module,
              external_dataload: DataLoader,
              test_grad,
              device
              ):
    dev = device
    test_grad = torch.tensor(test_grad).to(dev)
    # model.eval()
    model.to(dev)
    score_list = []
    index_list = []
    for i, batch in tqdm(enumerate(external_dataload), desc="--external_data", total=len(external_dataload)):
        model.zero_grad()
        # with torch.no_grad():
        index =  batch.pop("id", None)
        cls_labels = batch.pop("cls_labels", None)
        for k,v in batch.items():
            batch[k] = v.to(dev)
        outputs = model(**batch)
        outputs.loss.backward()

        weight_grad = model.bert.encoder.layer[11].attention.output.dense.weight.grad.view((1,-1))
        # bias_grad = model.bert.encoder.layer[11].attention.output.dense.bias.grad.view(-1)
        score = torch.mm(weight_grad,test_grad.T)

        score_list.append(score.cpu().numpy().tolist())
        index_list.append(index.cpu().numpy().tolist())
        # if i == 5:
        #     model.zero_grad()
        #     break
    return score_list, index_list

