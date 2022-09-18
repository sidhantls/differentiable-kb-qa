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
from utils.dataset_utils import QADataset2

import models_twohop

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pdb

parser = argparse.ArgumentParser("metaqa-reifiedkb")

parser.add_argument("--num_hops", help="num_hops dataset to train on MetaQA. 1, 2 or 3", type=int, required=False, default=2)
args = parser.parse_args()

NUM_HOPS = args.num_hops

assert NUM_HOPS >= 2

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
    param.requires_grad = True
    
# for layer in trans_model.encoder.layer[4].parameters():
#     layer.requires_grad_ = True

# for param in trans_model.encoder.layer[5].parameters():
#     param.requires_grad = True
    
# for param in trans_model.pooler.parameters(): 
#     param.requires_grad = True

trans_model.train()
trans_size=384

test_pairs_hop = []
train_pairs = []
val_pairs = [] 

for hop_num in range(1, NUM_HOPS+1):
    qa_paths = list((data_dir/f'{hop_num}-hop/vanilla').glob('*.txt'))


    qa_test = list(filter(lambda x: 'test' in x.name, qa_paths))[0]
    qa_val = list(filter(lambda x: 'dev' in x.name, qa_paths))[0]
    qa_train = list(filter(lambda x: 'train' in x.name, qa_paths))[0]

    # ntm data
    if USE_NTM_TRAINING:
        qa_paths = (data_dir/f'{hop_num}-hop/ntm').glob('*.txt')

    
    pairs = data_utils.load_qa_pairs(qa_val, hop_num)
    print(f'Hop: {hop_num}, num val pairs {len(pairs)}')
    val_pairs += pairs 

    pairs = data_utils.load_qa_pairs(qa_train, hop_num)
    print(f'Hop: {hop_num}, num train pairs {len(pairs)}')
    train_pairs += pairs 

    pairs = data_utils.load_qa_pairs(qa_test, hop_num)
    print(f'Hop: {hop_num}, num test pairs {len(pairs)}\n')
    test_pairs_hop.append(pairs)

print('Num train pairs ', len(train_pairs))
print('Num val pairs ', len(val_pairs))
print('Total test pairs ', sum([len(pair) for pair in test_pairs_hop]))
# data_utils.santity_check(val_pairs, entity_to_idx)
# data_utils.santity_check(test_pairs, entity_to_idx)
# data_utils.santity_check(train_pairs, entity_to_idx)

# # Tokenize sentences
val_tokens = tokenizer([row[0] for row in val_pairs], padding=True, truncation=True, max_length=200, return_tensors='pt')
train_tokens = tokenizer([row[0] for row in train_pairs], padding=True, truncation=True, max_length=200, return_tensors='pt')

val_dataset = QADataset2(val_pairs, val_tokens, entity_to_idx)
train_dataset = QADataset2(train_pairs, train_tokens, entity_to_idx)

train_bs = 128
eval_bs = 256
val_dl = DataLoader(val_dataset, batch_size=eval_bs)
train_dl = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)

test_dls = []
for test_pairs in test_pairs_hop:
    test_tokens = tokenizer([row[0] for row in test_pairs], padding=True, truncation=True, max_length=200, return_tensors='pt')
    test_dataset = QADataset2(test_pairs, test_tokens, entity_to_idx)
    test_dl = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False)
    test_dls.append(test_dl)

for batch in train_dl:
    break

subject_matrix, rel_matrix, object_matrix = data_utils.create_differentiable_kg(triplets, entity_to_idx, relation_to_idx)
object_matrix = torch.transpose(object_matrix, 0, 1)


if NUM_HOPS == 1 or NUM_HOPS == 2 or NUM_HOPS == 3:
    net = models_twohop.KBLightning(trans_model, subject_matrix, rel_matrix, object_matrix, NUM_HOPS, trans_output_size=trans_size, num_training_steps=len(train_dl)*4)
else:
    raise ValueError('Expected num hops to be 1, 2, or 3')
  
max_epochs = 6

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

# raise ValueError()

log_every_n_steps = 10
logger = TensorBoardLogger("unified_kblogs", name=f"unified-reified-{NUM_HOPS}-fulltrans", flush_secs=20)

trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, logger=logger, callbacks=[early_stop_callback, checkpoint_callback],
                    val_check_interval=0.50, log_every_n_steps=log_every_n_steps)
trainer.fit(net, train_dl, val_dl)

# load best checkpoint for testing
# CKPT_PATH = trainer.checkpoint_callback.best_model_path
# print('Model saved path ', CKPT_PATH)

# net = net.load_from_checkpoint(CKPT_PATH, trans_model=trans_model, subject_matrix=subject_matrix, rel_matrix=rel_matrix, object_matrix=object_matrix, num_hops=NUM_HOPS, trans_output_size=trans_size)


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
    


for num_hop, test_dl in enumerate(test_dls): 
    res = trainer.test(net, test_dl)
    res = res[0]
    res = {f'{key}_{num_hop+1}hop': res[key] for key in res if 'hit' in key or 'acc' in key}
    print(res)
    logger.log_metrics(res)

    
