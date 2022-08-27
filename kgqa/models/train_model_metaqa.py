import os
import sys 
from pathlib import Path 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import argparse 

root_dir = Path(os.getcwd()).parents[0]
sys.path.append(str(root_dir))

from utils import data_utils
from utils.dataset_utils import QADataset

import models_nhop

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


parser = argparse.ArgumentParser("metaqa-reifiedkb")

parser.add_argument("--num_hops", help="num_hops dataset to train on MetaQA. 1, 2 or 3", type=int, required=True)
args = parser.parse_args()

NUM_HOPS = args.num_hops
USE_NTM_TRAINING = False # rephrased training dataset of metaqa

root_dir = Path(os.getcwd()).parents[0]
sys.path.append(str(root_dir))

data_dir = root_dir/'datasets/MetaQA'

kg_path = data_dir/'kb.txt'

os.listdir(data_dir)
assert kg_path.exists()

triplets, entity_to_idx, relation_to_idx, idx_to_entity, idx_to_relation = data_utils.load_triplets_metaqa(kg_path)


# # Load model from HuggingFace Hub
# MODEL_NAME = 'microsoft/MiniLM-L12-H384-uncased'
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# trans_model = AutoModel.from_pretrained(MODEL_NAME)
# for param in trans_model.parameters():
#     param.requires_grad = True

# This model is more resource effective. Negligible difference in final score 
# Fewer parameters to train and smaller model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
trans_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
for param in trans_model.parameters():
    param.requires_grad = False
    
for layer in trans_model.encoder.layer[4].parameters():
    layer.requires_grad_ = True

for param in trans_model.encoder.layer[5].parameters():
    param.requires_grad = True
    
for param in trans_model.pooler.parameters(): 
    param.requires_grad = True

trans_model.train()
trans_size=384

qa_dir = data_dir/f'{NUM_HOPS}-hop/vanilla'
qa_paths = os.listdir(qa_dir)
qa_test = list(filter(lambda x: 'test' in x, qa_paths))[0]
qa_val = list(filter(lambda x: 'dev' in x, qa_paths))[0]

# ntm data
if USE_NTM_TRAINING:
    qa_dir = data_dir/f'{NUM_HOPS}-hop/ntm'
    qa_paths = os.listdir(qa_dir)

qa_train = list(filter(lambda x: 'train' in x, qa_paths))[0]

val_pairs = data_utils.load_qa_pairs(qa_dir/qa_val)
train_pairs = data_utils.load_qa_pairs(qa_dir/qa_train)
test_pairs = data_utils.load_qa_pairs(qa_dir/qa_test)

data_utils.santity_check(val_pairs, entity_to_idx)
data_utils.santity_check(test_pairs, entity_to_idx)
data_utils.santity_check(train_pairs, entity_to_idx)

# # Tokenize sentences
val_tokens = tokenizer([row[0] for row in val_pairs], padding=True, truncation=True, max_length=200, return_tensors='pt')
train_tokens = tokenizer([row[0] for row in train_pairs], padding=True, truncation=True, max_length=200, return_tensors='pt')
test_tokens = tokenizer([row[0] for row in test_pairs], padding=True, truncation=True, max_length=200, return_tensors='pt')

val_dataset = QADataset(val_pairs, val_tokens, entity_to_idx)
test_dataset = QADataset(test_pairs, test_tokens, entity_to_idx)
train_dataset = QADataset(train_pairs, train_tokens, entity_to_idx)

train_bs = 128
eval_bs = 256
val_dl = DataLoader(val_dataset, batch_size=eval_bs)
train_dl = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=eval_bs)

for batch in train_dl:
    break

subject_matrix, rel_matrix, object_matrix = data_utils.create_differentiable_kg(triplets, entity_to_idx, relation_to_idx)
object_matrix = torch.transpose(object_matrix, 0, 1)


if NUM_HOPS == 1 or NUM_HOPS == 2 or NUM_HOPS == 3:
    net = models_nhop.KBLightning(trans_model, subject_matrix, rel_matrix, object_matrix, NUM_HOPS, trans_output_size=trans_size, num_training_steps=len(train_dl)*4)
else:
    raise ValueError('Expected num hops to be 1, 2, or 3')
  
max_epochs = 5

USE_GPU = torch.cuda.is_available()

early_stop_callback = EarlyStopping(monitor="val_hit_k1", 
                                    min_delta=0.003,
                                    patience=2, 
                                    verbose=False, 
                                    mode="max")

checkpoint_callback = ModelCheckpoint(monitor=f"val_hit_k1",
                                    save_top_k=1,
                                      dirpath='checkpoints',
                                      mode='max',
                                    )

callbacks=[early_stop_callback, checkpoint_callback]

if USE_GPU:
    net.object_matrix = net.object_matrix.cuda()
    net.subject_matrix = net.subject_matrix.cuda()
    net.rel_matrix = net.rel_matrix.cuda()
    gpus = 1 
else:
    net = net.cpu()
    net.object_matrix = net.object_matrix.cpu()
    net.subject_matrix = net.subject_matrix.cpu()
    net.rel_matrix = net.rel_matrix.cpu()
    gpus = 0
    
log_every_n_steps = 10
logger = TensorBoardLogger("kb_logs", name=f"reifedkb-{NUM_HOPS}")

trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, logger=logger, callbacks=[early_stop_callback, checkpoint_callback],
                    val_check_interval=0.50, log_every_n_steps=log_every_n_steps)
trainer.fit(net, train_dl, val_dl)

# load best checkpoint for testing
CKPT_PATH = trainer.checkpoint_callback.best_model_path


net = net.load_from_checkpoint(CKPT_PATH, trans_model=trans_model, subject_matrix=subject_matrix, rel_matrix=rel_matrix, object_matrix=object_matrix, num_hops=NUM_HOPS, trans_output_size=trans_size)


if USE_GPU:
    net.object_matrix = net.object_matrix.cuda()
    net.subject_matrix = net.subject_matrix.cuda()
    net.rel_matrix = net.rel_matrix.cuda()
    gpus = 1 
else:
    net = net.cpu()
    net.object_matrix = net.object_matrix.cpu()
    net.subject_matrix = net.subject_matrix.cpu()
    net.rel_matrix = net.rel_matrix.cpu()
    gpus = 0
    
trainer.test(net, test_dl)
print('Model saved path ', CKPT_PATH)
