import torch

class Config:
    def __init__(self, args) -> None:
        # data hyperparameter
        self.batch_size = args.batch_size
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.max_seq_length = args.max_seq_length
        self.num_labels = args.num_labels

        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.external_file = args.external_file

        self.gpus = args.gpus
        if args.device_ids != None:
            self.device_ids = [int(ids) for ids in args.device_ids]

        # optimizer hyperparameter
        self.learning_rate = args.learning_rate
        self.max_grad_norm = args.max_grad_norm

        self.mode = args.mode
        # training
        self.device = torch.device(args.device)
        self.num_epochs = args.num_epochs
        self.early_stop = args.early_stop
        self.num_process = args.num_process
        self.mlm_weight = args.mlm_weight

        # model
        self.model_folder = args.model_folder
        self.bert_model_name = args.bert_model_name
        self.bert_folder = args.bert_folder

        self.fp16 = args.fp16
