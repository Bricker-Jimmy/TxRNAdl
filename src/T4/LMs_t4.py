# %%
import re
import os
import json
import random

from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torchvision.ops.focal_loss import sigmoid_focal_loss


class LMsModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        pass

    def load_model(self):
        return self

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def tokenizer(self, batchs):
        return self.tokenizer(batchs, return_tensors='pt')


class HuggingFaceLMsModel(LMsModel):
    def __init__(
        self,
        model_path,
        lms_class=PreTrainedModel,
        tokenizer_class=PreTrainedTokenizer
    ):
        super().__init__()
        self.model_path = model_path
        self.lms_class = lms_class
        self.tokenizer_class = tokenizer_class
        pass

    def load_model(self):
        super().load_model()
        self.model = self.lms_class.from_pretrained(
            self.model_path, output_hidden_states=True)
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.model_path, do_lower_case=False)
        return self


class ProtBert_LMsModel(HuggingFaceLMsModel):
    def __init__(self, model_path, lms_class=BertModel, tokenizer_class=BertTokenizer):
        super().__init__(model_path=model_path,
                         lms_class=lms_class, tokenizer_class=tokenizer_class)
        pass

    def preprocess(sequence: str):
        sequence = " ".join(sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        return sequence


class DNABert_LMsModel(ProtBert_LMsModel):
    def __init__(self, model_path, lms_class=BertModel, tokenizer_class=BertTokenizer):
        super().__init__(model_path=model_path,
                         lms_class=lms_class, tokenizer_class=tokenizer_class)
        pass

    def preprocess(sequence: str):
        sequence = " ".join([
            token if len(re.findall('((?![ATCG]).)+', token)) == 0 else '[UNK]'
            for token in [
                sequence[index: index + 6]
                for index in range(len(sequence) - 6)
            ]
        ])
        return sequence


class DNA_Binary_Fasta_Dataset(Dataset):

    def __init__(
        self,
        pos_seq_list,
        neg_seq_list,
        transform=None,
        target_transform=None,
        cache=False
    ):
        # Do tokenize in function of transform.
        # target is nothing to do.

        super().__init__()
        self.transform = transform
        self.target_transform = target_transform

        self.pos_seq_list = pos_seq_list
        random.shuffle(self.pos_seq_list)
        self.neg_seq_list = neg_seq_list
        random.shuffle(self.neg_seq_list)

        self.cache = [None, ] * len(self) if cache == True else None
        pass

    def __len__(self):
        return len(self.pos_seq_list) + len(self.neg_seq_list)

    def __getitem__(self, idx: int):
        if self.cache is not None and self.cache[idx] is not None:
            return self.cache[idx]

        if idx < len(self.pos_seq_list):
            seq = str(self.pos_seq_list[idx].seq)
            label = 1
        else:
            seq = str(self.neg_seq_list[idx - len(self.pos_seq_list)].seq)
            label = 0

        seq = seq[:300]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)

        if self.cache is not None:
            self.cache[idx] = (seq, label)
            return self.cache[idx]
        return (seq, label)


def train_loop(
    train_data: DataLoader,
    model: nn.Module,
    loss_fn,
    optimizer: optim.Optimizer,
    device: str,
    loadmodel: HuggingFaceLMsModel,
    step: int,
):
    model.train()
    loadmodel.model.train()

    total_loss = 0
    batch_num = 0
    pred_list = []
    target_list = []
    for i, batch in enumerate(train_data):
        batch_num += 1
        seq, targets = batch
        seq = seq.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        seq = loadmodel(torch.squeeze(seq, dim=1))[1]

        pred = model(seq)

        loss = loss_fn(pred, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        local_averaged_loss = loss.item() / len(batch)
        pred_list.extend(pred.detach().cpu().numpy())
        target_list.extend(targets)
        total_loss += local_averaged_loss
    total_loss /= batch_num
    return total_loss, pred_list, target_list


def test_loop(
    test_data: DataLoader,
    model: nn.Module,
    loss_fn,
    device: str,
    loadmodel: HuggingFaceLMsModel,
    step: int,
):
    model.eval()
    loadmodel.model.eval()

    batch_num = 0
    total_loss = 0
    pred_list = []
    target_list = []
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            batch_num += 1
            seq, targets = batch
            seq = seq.to(device)
            targets = targets.to(device)

            seq = loadmodel(torch.squeeze(seq, dim=1))[1]

            pred = model(seq)

            local_averaged_loss = loss_fn(
                pred, targets).item() / len(batch)
            pred_list.extend(pred.detach().cpu().numpy())
            target_list.extend(targets)
            total_loss += local_averaged_loss

    total_loss /= batch_num
    return total_loss, pred_list, target_list


class Lr(nn.Module):
    def __init__(self, input_seq_len, input_seq_dim):
        super(Lr, self).__init__()
        input_dim = input_seq_dim
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

    def forward(self, seq):
        x = torch.squeeze(seq, dim=1)
        logits = self.output_layer(x)
        return logits


def train_func(args: dict, data_config_obj: dict, model_config_obj: dict):

    # load data
    train_pos_seq_list = data_config_obj['train_pos_seq_list']
    train_neg_seq_list = data_config_obj['train_neg_seq_list']
    test_pos_seq_list = data_config_obj['test_pos_seq_list']
    test_neg_seq_list = data_config_obj['test_neg_seq_list']

    calculate_device = args['training_config']['device']
    if os.getenv('DEBUG') is not None:
        calculate_device = calculate_device['debug']
    else:
        calculate_device = calculate_device['other']

    model_path = args['model_path'][model_config_obj['lmmodel_config']
                                    ['name']]['local']
    if os.path.exists(model_path) == False:
        model_path = args['model_path'][model_config_obj['lmmodel_config']
                                        ['name']]['huggingface']

    loadmodel = model_config_obj['lmmodel_config']['class'](
        model_path=model_path).load_model()

    loadmodel.model.to(calculate_device)

    def seq_transform_func(seq):
        # loadmodel.model.eval()
        return loadmodel.tokenizer.encode(model_config_obj['lmmodel_config']['class'].preprocess(seq), max_length=args['input_seq_len'], padding="max_length", truncation=True, return_tensors='pt')
        # return torch.Tensor(loadmodel(seq).last_hidden_state.detach().numpy())

    def target_transform_func(target):
        return torch.Tensor([target, ])

    train_data = DNA_Binary_Fasta_Dataset(
        train_pos_seq_list,
        train_neg_seq_list,
        transform=seq_transform_func,
        target_transform=target_transform_func
    )
    test_data = DNA_Binary_Fasta_Dataset(
        test_pos_seq_list,
        test_neg_seq_list,
        transform=seq_transform_func,
        target_transform=target_transform_func
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args['training_config']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(
        test_data, batch_size=args['training_config']['batch_size'], shuffle=True)

    model = Lr(
        input_seq_len=args['input_seq_len'],
        input_seq_dim=args['model_path'][model_config_obj['lmmodel_config']['name']]['dim']
    )

    model.to(calculate_device)
    loss_fn = functools.partial(sigmoid_focal_loss, reduction='mean')
    # loss_fn.to(calculate_device)
    epochs = args['training_config']['epochs']
    optimizer = model_config_obj['optimer_config']['class'](
        model.parameters(),
        lr=args['training_config']['learning_rate'],
        # momentum=args['training_config']['momentum']
    )

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        if t != 0:

            a = train_loop(train_dataloader, model, loss_fn,
                           optimizer, calculate_device, loadmodel, t)
            b = test_loop(test_dataloader, model, loss_fn,
                          calculate_device, loadmodel, t)
            # ! 注意，这里pred和target写反了。
            train_pred = torch.Tensor(a[2]).cpu().numpy()
            train_target = a[1]

            test_pred = torch.Tensor(b[2]).cpu().numpy()
            test_target = b[1]

            os.makedirs(os.path.join(
                args['training_config']['save_path'], args['githash'], args['taskname']), exist_ok=True)
            with open(os.path.join(args['training_config']['save_path'], args['githash'], args['taskname'], f"{t}.json"), "w+", encoding="UTF-8") as f:
                json.dump({
                    "train_target": train_pred.tolist(),
                    "train_pred": [i[0].item() for i in train_target],
                    "test_target": test_pred.tolist(),
                    "test_pred": [i[0].item() for i in test_target],
                }, f)

        else:
            b = test_loop(test_dataloader, model, loss_fn,
                          calculate_device, loadmodel, t)


# %%
import numpy as np
import torch
from torch import nn, optim
import os
import yaml
import itertools
import functools
import git
import json

config_file = 'config/config.yml'

githash = "T4"
args_template = None
with open(config_file, 'r', encoding='utf-8') as f:
    args_template = yaml.load(f, Loader=yaml.FullLoader)


random_seed = args_template['randseed']
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

model_class_list = [
    {
        "class": DNABert_LMsModel,
        'name': 'dnabert6'
    },
]

optim_class_list = [
    {
        "class": functools.partial(optim.SGD, momentum=0.9),
        'name': 'SGD'
    }
]

from Bio import SeqIO
# %%
for lmmodel_config, optimer_config in itertools.product(model_class_list, optim_class_list):

    args_template['githash'] = githash
    args_template['training_config']['optimizier'] = optimer_config['name']
    args_template['training_config']['model'] = lmmodel_config['name']

    ################### data ########################

    data_config = {
        "train_pos_seq_list": list(
            SeqIO.parse(
                'data/T4_pos_filter.fasta', 'fasta'
            )
        )[0:250],
        "train_neg_seq_list": list(
            SeqIO.parse(
                'data/T4_neg_filter.fasta', 'fasta'
            )
        )[0:13116],
        "test_pos_seq_list": list(
            SeqIO.parse(
                'data/T4_pos_filter.fasta', 'fasta'
            )
        )[250:],
        "test_neg_seq_list": list(
            SeqIO.parse(
                'data/T4_neg_filter.fasta', 'fasta'
            )
        )[13116:]
    }

    #################################################

    args_template['taskname'] = f"{lmmodel_config['name']}_{optimer_config['name']}"

    train_func(
        args=args_template,
        data_config_obj=data_config,
        model_config_obj={
            "lmmodel_config": lmmodel_config,
            "optimer_config": optimer_config
        }
    )
