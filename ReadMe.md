# Farewell to Aimless Large-scale Pretraining: Influential Subset Selection for Language Model

- This repo releases our implementation for ISS.
- It is built based on the scratch Bert model, and finetuned on our data.

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.26.1)

You can install the required libraries by running 

```
pip install -r requirements.txt
```


## Data

The data used in our model is based on the data selected by BM25, and the ISS method is selected on it.

You can download the data from [Huggingface](https://huggingface.co/yxchar).



## Training

First, you need to finetune a model using an existing dataset. This will help the ISS method better identify the data convergence direction.
```
cd ISS
bash train_score_model.sh
```

Then, use the ISS method to select the most relevant unsupervised data for the downstream task based on scores from unsupervised data.
```
cd ISS
bash get_iss_data.sh
python select_data_byscore_for.py
```

Use the filtered unsupervised dataset and the downstream task dataset to perform the Bert model's pretraining tasks, following the TLM method.
```
cd pretrain
bash example_scripts/train.sh
```

## Finetune
Finally, use the pretrained model for finetuning on the downstream task.
```
cd pretrain
bash example_scripts/finetune.sh
```


## Citation
```latex
@article{Wang_Zhou_Zhang_Zhou_Gao_Wang_Zhang_Gao_Chen_Gui_2023,
    title={Farewell to Aimless Large-scale Pretraining: Influential Subset Selection for Language Model},
    author={Wang, Xiao and Zhou, Weikang and Zhang, Qi and Zhou, Jie and Gao, Songyang and Wang, Junzhe and Zhang, Menghan and Gao, Xiang and Chen, Yunwen and Gui, Tao},
    year={2023},
    month={May},
    language={en-US}
}
```


