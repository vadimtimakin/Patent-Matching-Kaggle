# IMPORTS

import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import  Dataset

from bitsandbytes.optim import Adam8bit
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModel

# CONFIGURATION

class CFG:
    seed = 0xFACED

    path_to_csv = '/home/toefl/K/ptp/dataset/train.csv'
    output_dir = '/home/toefl/K/ptp/logs'

    model_path = "microsoft/deberta-v3-large"
    save_dir = '/home/toefl/K/ptp/checkpoints/deberta-v3-large-plus-plus'

    num_fold = 10
    exec_fold = [5, 6, 7, 8, 9]
    
    learning_rate = 1e-5
    eta_min = 1e-8
    weight_decay = 0.01
    epochs = 10
    batch_size = 4
    dropout=0.1

# SEED

def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(CFG.seed)

# DATASET

class PatentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.inputs = df['input'].values.astype(str)
        self.targets = df['target'].values.astype(str)
        self.label = df['score'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = self.inputs[item]
        targets = self.targets[item]
        label = self.label[item]
        
        return {
        **self.tokenizer(inputs, targets),
        'label':label.astype(np.float32)
        }

# MODEL

class LesGoNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(LesGoNet, self).__init__()
        self.config = AutoConfig.from_pretrained(CFG.model_path)
        self.model_name = model_name
        
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(self.config)

        self.fc_dropout = nn.Dropout(CFG.dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            try:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            except AttributeError:
                module.weight.data.normal_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            try:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            except AttributeError:
                module.weight.data.normal_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        logits = self.fc(self.fc_dropout(feature))
        
        out_dict = {'logits' : logits.squeeze(-1)}
        return out_dict

# CUSTOM FUNCTIONS

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }


class RegressionTrainerBCE(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(inputs)
        logits = outputs.get('logits')
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

# INITIALIZING CONSTANT OBJECTS

if not os.path.exists(CFG.save_dir):
    os.makedirs(CFG.save_dir, exist_ok=True)

oof_df = pd.DataFrame()

tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

train_df = pd.read_csv(CFG.path_to_csv)

args = TrainingArguments(
    output_dir=CFG.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CFG.learning_rate,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    num_train_epochs=CFG.epochs,
    weight_decay=CFG.weight_decay,
    metric_for_best_model="pearson",
    load_best_model_at_end=True,
)

# TRAINING

for fold in range(CFG.num_fold):
    if fold in CFG.exec_fold:
        print(f'Fold: {fold}')

        # Getting the data Data
        tr_data = train_df[train_df['kfold']!=fold].reset_index(drop=True)
        va_data = train_df[train_df['kfold']==fold].reset_index(drop=True)
        tr_dataset = PatentDataset(tr_data, tokenizer)
        va_dataset = PatentDataset(va_data, tokenizer)

        # Initializing the objects
        model = LesGoNet(CFG.model_path)
        optimizer = Adam8bit(model.parameters(), lr=CFG.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.eta_min)
        optimizers = optimizer, scheduler
        
        # Initializing the Trainer
        trainer = RegressionTrainerBCE(
            model,
            args,
            train_dataset=tr_dataset,
            eval_dataset=va_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
        )

        # Train!
        trainer.train()
        trainer.save_model(os.path.join(CFG.save_dir, f'fold_{fold}'))

        # Compute OOF
        outputs = trainer.predict(va_dataset)
        predictions = outputs.predictions.reshape(-1)
        va_data['preds'] = predictions
        oof_df = pd.concat([oof_df, va_data])

# PRINT AND SAVE OOF

predictions = oof_df['preds'].values
label = oof_df['score'].values
eval_pred = predictions, label
print(compute_metrics(eval_pred))
oof_df.to_csv(os.path.join(CFG.save_dir, 'oof_df.csv'))