import torchmetrics
from pytorch_lightning import LightningModule
import pytorch_lightning
import torch
from torch import nn
import transformers
from torch.functional import F


def calculate_accuracy(preds, true):    
    pred_class = preds.argmax(-1)

    return (pred_class == true).sum()/len(true)

class KBLightning(LightningModule):
    """
    Pytorch Lightning model for question answering based on differentiable knowledge bases 

    Answer questions over num_hops number of hops

    """

    def __init__(self, trans_model, subject_matrix, rel_matrix, object_matrix, num_hops, trans_output_size=384, num_training_steps=100):
        super(KBLightning, self).__init__()
        """
        Expects: 

        trans_model: hugginface transformer model 
        subject_matrix: (num_triplets, num_entities) sparse
        rel_matrix: (num_triplets, num_rels) sparse
        object_matrix: (num_triplets, num_entities) sparse
        """

        self.num_entities = subject_matrix.shape[1]
        self.num_rels = rel_matrix.shape[1]

        self.trans_model = trans_model

        self.num_hops = num_hops
        self.decoders = nn.ModuleList()
        
        for idx in range(self.num_hops):
            self.decoders.append(nn.Linear(trans_output_size, self.num_rels))

        self.subject_matrix = subject_matrix 
        self.rel_matrix = rel_matrix 
        self.object_matrix = object_matrix

        ## checks
        if torch.is_tensor(self.subject_matrix) and self.subject_matrix.shape[0] != self.rel_matrix.shape[0]:
            raise ValueError(f'Unexpected shape of subject_matrix or relation matrix. Expected dimension 0 to be same in subject_matrix and rel_matrix')

        # required for LR scheduler
        self.num_training_steps = num_training_steps

    def forward(self, trans_input, subject_vectors):
        model_output = self.trans_model(**trans_input)

        encoded_relations = self.get_cls_hidden(model_output)

        curr_subject_vectors = subject_vectors
        for idx in range(self.num_hops):
            pred_logits = self.decoders[idx](encoded_relations)
            pred_relations = torch.softmax(pred_logits, 1)
            predicted_objects = self.follow(curr_subject_vectors, pred_relations, self.subject_matrix, self.rel_matrix, self.object_matrix)
            curr_subject_vectors = torch.transpose(predicted_objects, 0, 1)

        return predicted_objects

    def follow(self, subject, relation, subject_matrix, rel_matrix, object_matrix):
        """
        Performs the follow operation as mentioned in paper.

        Expects: 

        subject: (num_entities, bs) sparse
        subject_matrix: (num_triplets, num_entities) sparse
        relation: (bs, num_rels) dense 
        rel_matrix: (num_triplets, num_rels) sparse
        object_matrix: (num_entities, num_triplets) sparse
        
        """
        assert subject.shape[0] == self.num_entities

        subject_vectors = torch.sparse.mm(subject_matrix, subject) # (num_triplets, bs)

        relation = torch.transpose(relation, 0, 1)

        relation_vectors = torch.sparse.mm(rel_matrix, relation) # (num_triplets, bs)

        object_query = subject_vectors * relation_vectors

        predicted_objects = torch.sparse.mm(object_matrix, object_query) # (num_entities, bs)
        predicted_objects = torch.transpose(predicted_objects, 0, 1) # (bs, num_entities)
        
        return predicted_objects

    def get_cls_hidden(self, model_output):
        cls_hidden_state = model_output['last_hidden_state'][:, 0, :]

        return cls_hidden_state

    def shared_step(self, batch, dataname='train'):
        trans_input, subject_vector, object_labels = batch
        subject_vector2 = torch.transpose(subject_vector, 0, 1)
        
        object_logits = self(trans_input, subject_vector2)
        object_logits_thresh = torch.clamp(object_logits, max=1., min=0.)


        loss = calculate_BCE(object_logits_thresh, object_labels)

        if dataname == 'train':
            on_step = True
        else:
            on_step = False
            
        with torch.no_grad():
            object_labels = object_labels.detach().cpu()
            object_preds = object_logits.detach().cpu()

            hit_k1 = get_hit_k1(object_preds, object_labels)           

            self.log(f'{dataname}_hit_k1', hit_k1, on_step=on_step, on_epoch=True, prog_bar=True)
    
        self.log(f'{dataname}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True)
        
        return loss
    
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='train')
        
        return loss


    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='test')

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='val')

        return loss

    def predict_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='val')

        return loss

    
    def configure_optimizers(self):
        optimizer = transformers.AdamW(lr=5e-3, params=self.parameters())
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.num_training_steps//4, self.num_training_steps, last_epoch = -1)
        
        return optimizer

"""
Below is just experimentation 
"""

class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        self.hidden_size = hidden_size
        init_tensor = torch.randn(input_size+hidden_size, hidden_size)
        self.Wz = torch.nn.Parameter(init_tensor.clone(), True)
        self.Wr = torch.nn.Parameter(init_tensor.clone(), True)
        self.W = torch.nn.Parameter(init_tensor.clone(), True)
    
    def forward(self, x, h=None):
        if isinstance(h, type(None)):
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            
        hx = torch.cat([h,x], dim=1)
        
        zt = torch.mm(hx, self.Wz)
        zt = torch.sigmoid(zt)
        
        rt = torch.mm(hx, self.Wr)
        rt = torch.sigmoid(rt)
        
        fuse = rt * zt
        htt = torch.mm(torch.cat([fuse, x], 1), self.W)
        htt = torch.tanh(htt)
        
        ht = (1-zt) * h + zt * htt
        
        return ht


class GNNLightning2(LightningModule):
    """
    Pytorch lightning trainer
    """

    def __init__(self, trans_model, subject_matrix, rel_matrix, object_matrix, num_hops, trans_output_size=384):
        super(GNNLightning2, self).__init__()
        """
        Expects: 

        trans_model: hugginface transformer model 
        subject_matrix: (num_triplets, num_entities) sparse
        rel_matrix: (num_triplets, num_rels) sparse
        object_matrix: (num_triplets, num_entities) sparse
        """

        self.num_entities = subject_matrix.shape[1]
        self.num_rels = rel_matrix.shape[1]

        self.trans_model = trans_model

        self.num_hops = num_hops
        self.decoders = nn.ModuleList()
        
        self.rnn = GRUCell(input_size=trans_output_size, hidden_size=trans_output_size)
        self.decoder = nn.Linear(trans_output_size, self.num_rels)

        self.subject_matrix = subject_matrix 
        self.rel_matrix = rel_matrix 
        self.object_matrix = object_matrix

        ## checks
        if torch.is_tensor(self.subject_matrix) and self.subject_matrix.shape[0] != self.rel_matrix.shape[0]:
            raise ValueError(f'Unexpected shape of subject_matrix or relation matrix. Expected dimension 0 to be same in subject_matrix and rel_matrix')


    
    def forward(self, trans_input, subject_vectors):
        model_output = self.trans_model(**trans_input)

        encoded_relations = self.get_cls_hidden(model_output)

        curr_subject_vectors = subject_vectors
        next_hidden = None
        for idx in range(self.num_hops):
            next_hidden = self.rnn(encoded_relations, next_hidden)
            pred_logits = self.decoder(next_hidden)
            pred_relations = torch.softmax(pred_logits, 1)
            predicted_objects = self.follow(curr_subject_vectors, pred_relations, self.subject_matrix, self.rel_matrix, self.object_matrix)
            curr_subject_vectors = torch.transpose(predicted_objects, 0, 1)

        return predicted_objects     

    def follow(self, subject, relation, subject_matrix, rel_matrix, object_matrix):
        """
        Performs the follow operation as mentioned in paper.

        Expects: 

        subject: (num_entities, bs) sparse
        subject_matrix: (num_triplets, num_entities) sparse
        relation: (bs, num_rels) dense 
        rel_matrix: (num_triplets, num_rels) sparse
        object_matrix: (num_entities, num_triplets) sparse
        
        """
        assert subject.shape[0] == self.num_entities

        subject_vectors = torch.sparse.mm(subject_matrix, subject) # (num_triplets, bs)

        relation = torch.transpose(relation, 0, 1)

        relation_vectors = torch.sparse.mm(rel_matrix, relation) # (num_triplets, bs)

        object_query = subject_vectors * relation_vectors

        predicted_objects = torch.sparse.mm(object_matrix, object_query) # (num_entities, bs)
        predicted_objects = torch.transpose(predicted_objects, 0, 1) # (bs, num_entities)
        
        return predicted_objects


    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_cls_hidden(self, model_output):
        cls_hidden_state = model_output['last_hidden_state'][:, 0, :]

        return cls_hidden_state

    
    def shared_step(self, batch, dataname='train'):
        trans_input, subject_vector, object_labels = batch
        subject_vector2 = torch.transpose(subject_vector, 0, 1)
        
        object_logits = self(trans_input, subject_vector2)
        object_logits_thresh = torch.clamp(object_logits, max=1., min=0.)


        loss = calculate_BCE(object_logits_thresh, object_labels)

        if dataname == 'train':
            on_step = True
        else:
            on_step = False
            
        with torch.no_grad():
            object_labels = object_labels.detach().cpu()
            object_preds = object_logits.detach().cpu()

            hit_k1 = get_hit_k1(object_preds, object_labels)           

            self.log(f'{dataname}_hit_k1', hit_k1, on_step=on_step, on_epoch=True, prog_bar=True)
    
        self.log(f'{dataname}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='train')
        
        return loss


    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='test')

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='val')

        return loss

    def predict_step(self, batch, batch_idx):
        loss = self.shared_step(batch, dataname='val')

        return loss

    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        optimizer = transformers.AdamW(lr=5e-3, params=self.parameters())

        return optimizer




def calculate_BCE(preds, labels):
    loss = F.binary_cross_entropy(preds, labels)
    return loss

# def compute_hit_1(preds, true):


def compute_multilabel_precision(preds, true, threshold=0.5):
    """
    Calculates multi-label precision and recall 
    
    Unused

    """

    preds = (preds>=threshold).float()

    true_positives = (preds == true) & (true == 1)
    recall = true_positives.sum(1)/true.sum(1)
    mean_recall = recall.mean()

    precision = true_positives.sum(1)/preds.sum(1)
    # fill nans and missing
    precision[precision==torch.inf] = torch.nan
    # precision[torch.isnan(precision)] = 0
    # mean_precision = precision.mean()

    mean_precision = precision.nanmean()

    f1 = mean_recall * mean_precision / (mean_precision + mean_recall)

    return mean_recall, mean_precision, f1


def get_hit_k1(preds, true, aggregate=True):
    """
    Calulates mean hit @ k = 1
    """

    preds_k1 = preds.argmax(1)
    
    # required in this format to use gather 
    pred_indices = torch.LongTensor([[pred.item()] for pred in preds_k1], device=preds.device)
    
    # get how many of these indices correspond with true value
    hits = true.gather(1, pred_indices)
    
    if aggregate:
        mean_hit = hits.mean()
    else:
        mean_hit = hits

    return mean_hit


