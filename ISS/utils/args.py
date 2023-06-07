import argparse

def parse_arguments(parser: argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu'] + ['cuda:' + str(i) for i in range(10)],
                        help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=4, help="default batch size is 10 (works well)")
    parser.add_argument('--max_seq_length', type=int, default=128, help="maximum sequence length")
    parser.add_argument('--train_num', type=int, default=40, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=20, help="The number of development data, -1 means all data")
    parser.add_argument('--num_labels', type=int, default=13, help="num_labels")

    parser.add_argument('--train_file', type=str, default=r"rain.csv")
    parser.add_argument('--dev_file', type=str, default=r'dev.csv')
    parser.add_argument('--test_file', type=str, default=r"test.csv")
    parser.add_argument('--external_file', type=str, default=r"small_external.csv")


    parser.add_argument('--gpus', type=bool, default=False)
    parser.add_argument('--device_ids', type=list, default='0',
                        help="Ids of GPU which you want to use to training e.g. 01234 ")
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # model
    parser.add_argument('--model_folder', type=str, default="bert-base",
                        help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="",
                        help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="bert-base-uncased", help="The bert model name to used")

    # training
    parser.add_argument('--mode', type=str, default="train",choices = ['train','tracIn'], help="training or tracIn")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=5, help="The number of epochs to run")
    parser.add_argument('--early_stop', type=int, default=3, help="The number of epochs to early stop")
    parser.add_argument('--mlm_weight', type=int, default=10, help="mlm weight")

    parser.add_argument('--fp16', type=int, default=0, choices=[0, 1], help="fp16")
    parser.add_argument('--num_process', type=int, default=1, help="multiple process tokenization")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args