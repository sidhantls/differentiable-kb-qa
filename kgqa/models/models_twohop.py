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
        

        hop_dim = trans_output_size//5

        dec1 = nn.Sequential(nn.Linear(trans_output_size+hop_dim, self.num_rels))
        self.decoders.append(dec1)

        for i in range(num_hops-1):
            dec2 = nn.Sequential(nn.Linear(trans_output_size+hop_dim+self.num_rels, self.num_rels))
            self.decoders.append(dec2)

        self.hoplayer1 =  nn.Sequential(nn.Linear(trans_output_size, hop_dim), nn.Tanh())
        self.hoplayer2 = nn.Sequential(nn.Linear(hop_dim, self.num_hops), nn.Tanh())
        self.hop_lossfn = nn.CrossEntropyLoss()

        self.subject_matrix = subject_matrix 
        self.rel_matrix = rel_matrix 
        self.object_matrix = object_matrix

        ## checks
        if torch.is_tensor(self.subject_matrix) and self.subject_matrix.shape[0] != self.rel_matrix.shape[0]:
            raise ValueError(f'Unexpected shape of subject_matrix or relation matrix. Expected dimension 0 to be same in subject_matrix and rel_matrix')

        # required for LR scheduler
        self.num_training_steps = num_training_steps

        self.layer_norm1 = nn.LayerNorm(trans_output_size+hop_dim)
        self.layer_norm2 = nn.LayerNorm(trans_output_size+hop_dim)

    def forward(self, trans_input, subject_vectors):
        model_output = self.trans_model(**trans_input)

        encoded_relations = self.get_cls_hidden(model_output)

        hop_hidden = self.hoplayer1(encoded_relations)
        hop_pred = self.hoplayer2(hop_hidden)

        curr_subject_vectors = subject_vectors

        # hop_hidden = torch.zeros(hop_hidden.shape, device=hop_hidden.device)

        for idx in range(self.num_hops):
            if idx == 0:
                input_dec = torch.cat([encoded_relations, hop_hidden], 1)
                # input_dec = self.layer_norm1(input_dec)
            elif idx == 1 or idx == 2:
                input_dec = torch.cat([encoded_relations, hop_hidden, pred_relations], 1)
                # input_dec = torch.cat([encoded_relations, torch.zeros(hop_hidden.shape, device=hop_hidden.device), pred_relations], 1)
                # input_dec = torch.cat([torch.zeros((encoded_relations.shape[0], encoded_relations.shape[1] +hop_hidden.shape[1]), device=hop_hidden.device), pred_relations], 1)
                # input_dec = self.layer_norm2(input_dec)
            else:
                raise ValueError('unaccounted case in model')

            pred_logits = self.decoders[idx](input_dec)
            pred_relations = torch.softmax(pred_logits, 1)

            predicted_objects = self.follow(curr_subject_vectors, pred_relations, self.subject_matrix, self.rel_matrix, self.object_matrix)
            curr_subject_vectors = torch.transpose(predicted_objects, 0, 1)

        return predicted_objects, hop_pred

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
        trans_input, subject_vector, object_labels, hop_labels = batch
        subject_vector2 = torch.transpose(subject_vector, 0, 1)
        
        object_logits, hop_pred = self(trans_input, subject_vector2)
        object_logits_thresh = torch.clamp(object_logits, max=1., min=0.)
        
        loss = calculate_BCE(object_logits_thresh, object_labels)
        loss2 = self.hop_lossfn(hop_pred, hop_labels)


        if dataname == 'train':
            on_step = True
            on_epoch = False
        else:
            on_step = False
            on_epoch = True
            
        with torch.no_grad():
            object_labels = object_labels.detach().cpu()
            object_preds = object_logits.detach().cpu()

            hit_k1 = get_hit_k1(object_preds, object_labels)           

            self.log(f'{dataname}_hit_k1', hit_k1, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
            self.log(f'{dataname}_loss1', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=False)
            self.log(f'{dataname}_loss2', loss2, on_step=on_step, on_epoch=on_epoch, prog_bar=False)


            preds = hop_pred.argmax(-1)
            acc = (hop_labels==preds).sum()/len(preds)
            self.log(f'{dataname}_hop_acc', acc, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

        # weighted loss for training 
        loss = loss + loss2/1000
        self.log(f'{dataname}_loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        
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
        optimizer = transformers.AdamW(lr=5e-4, params=self.parameters())
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.num_training_steps//4, self.num_training_steps, last_epoch = -1)
        
        return optimizer



class KBLightning2(LightningModule):
    """
    Pytorch Lightning model for question answering based on differentiable knowledge bases 

    Answer questions over num_hops number of hops

    """

    def __init__(self, trans_model, subject_matrix, rel_matrix, object_matrix, num_hops, trans_output_size=384, num_training_steps=100):
        super(KBLightning2, self).__init__()
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

        self.num_hops = 2
        self.decoders = nn.ModuleList()
        

        hop_dim = trans_output_size//5

        dec1 = nn.Sequential(nn.Linear(trans_output_size+hop_dim, self.num_rels))
        dec2 = nn.Sequential(nn.Linear(trans_output_size+hop_dim+self.num_rels, self.num_rels))

        self.decoders.append(dec1)
        self.decoders.append(dec2)

        self.hoplayer1 =  nn.Sequential(nn.Linear(trans_output_size, hop_dim), nn.Tanh())
        self.hoplayer2 = nn.Sequential(nn.Linear(hop_dim, self.num_hops), nn.Tanh())
        self.hop_lossfn = nn.CrossEntropyLoss()

        self.subject_matrix = subject_matrix 
        self.rel_matrix = rel_matrix 
        self.object_matrix = object_matrix

        ## checks
        if torch.is_tensor(self.subject_matrix) and self.subject_matrix.shape[0] != self.rel_matrix.shape[0]:
            raise ValueError(f'Unexpected shape of subject_matrix or relation matrix. Expected dimension 0 to be same in subject_matrix and rel_matrix')

        # required for LR scheduler
        self.num_training_steps = num_training_steps

        self.layer_norm1 = nn.LayerNorm(trans_output_size+hop_dim)
        self.layer_norm2 = nn.LayerNorm(trans_output_size+hop_dim)

    def forward(self, trans_input, subject_vectors):
        model_output = self.trans_model(**trans_input)

        encoded_relations = self.get_cls_hidden(model_output)

        hop_hidden = self.hoplayer1(encoded_relations)
        hop_pred = self.hoplayer2(hop_hidden)

        curr_subject_vectors = subject_vectors

        # hop_hidden = torch.zeros(hop_hidden.shape, device=hop_hidden.device)

        for idx in range(self.num_hops):
            
            if idx == 0:
                input_dec = torch.cat([encoded_relations, hop_hidden], 1)
                # input_dec = self.layer_norm1(input_dec)
            elif idx == 1 or idx == 2:
                input_dec = torch.cat([encoded_relations, hop_hidden, pred_relations], 1)
                # input_dec = self.layer_norm2(input_dec)
            else:
                raise ValueError(f'unaccounted case in model hop number {idx}')

            pred_logits = self.decoders[idx](input_dec)
            pred_relations = torch.softmax(pred_logits, 1)

            predicted_objects = self.follow(curr_subject_vectors, pred_relations, self.subject_matrix, self.rel_matrix, self.object_matrix)
            curr_subject_vectors = torch.transpose(predicted_objects, 0, 1)

        return predicted_objects, hop_pred

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
        trans_input, subject_vector, object_labels, hop_labels = batch
        subject_vector2 = torch.transpose(subject_vector, 0, 1)
        
        object_logits, hop_pred = self(trans_input, subject_vector2)
        object_logits_thresh = torch.clamp(object_logits, max=1., min=0.)
        
        loss = calculate_BCE(object_logits_thresh, object_labels)
        loss2 = self.hop_lossfn(hop_pred, hop_labels)


        if dataname == 'train':
            on_step = True
            on_epoch = False
        else:
            on_step = False
            on_epoch = True
            
        with torch.no_grad():
            object_labels = object_labels.detach().cpu()
            object_preds = object_logits.detach().cpu()

            hit_k1 = get_hit_k1(object_preds, object_labels)           

            self.log(f'{dataname}_hit_k1', hit_k1, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
            self.log(f'{dataname}_loss1', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=False)
            self.log(f'{dataname}_loss2', loss2, on_step=on_step, on_epoch=on_epoch, prog_bar=False)


            preds = hop_pred.argmax(-1)
            acc = (hop_labels==preds).sum()/len(preds)
            self.log(f'{dataname}_hop_acc', acc, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

        # weighted loss for training 
        loss = loss + loss2/50
        self.log(f'{dataname}_loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        
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
        optimizer = transformers.AdamW(lr=5e-4, params=self.parameters())
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, self.num_training_steps//4, self.num_training_steps, last_epoch = -1)
        
        return optimizer


def calculate_BCE(preds, labels):
    loss = F.binary_cross_entropy(preds, labels)
    return loss


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


def interpret_follow_output(out, idx_to_entity, threshold=0.5):
    """
    Interpret the follow result from the model
    """

    bs, num_entities = out.shape 
        
    outputs_names = []
    output_ids = []
    for idx in range(bs):
        object_probs = out[idx, :]
        
        # set all values less than threshold to be 0
        condition = object_probs >= threshold
        object_preds = object_probs.where(condition, torch.tensor(0.))
        # get indices where value is non zero (true preds)
        pred_object_ids = object_preds.nonzero(as_tuple=True)[0]
        
        pred_object_names = [idx_to_entity[object_id.item()] for object_id in pred_object_ids]
        outputs_names.append(pred_object_names)
        output_ids.append(pred_object_ids.tolist())
            
    return outputs_names, output_ids