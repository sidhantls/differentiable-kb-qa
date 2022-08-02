import torchmetrics
from pytorch_lightning import LightningModule
import pytorch_lightning
import torch
from torch import nn

def calculate_accuracy(preds, true):    
    pred_class = preds.argmax(-1)

    return (pred_class == true).sum()/len(true)

class GNNLightning(LightningModule):
    """
    Pytorch lightning trainer
    """

    def __init__(self, trans_model, subject_matrix, rel_matrix, object_matrix):
        super(GNNLightning, self).__init__()
        """
        Expects: 

        trans_model: hugginface transformer model 
        subject_matrix: (num_entities, num_triplets) sparse
        rel_matrix: (num_triplets, num_rels) sparse
        object_matrix: (num_triplets, num_entities) sparse
        """

        self.model = trans_model

        self.subject_matrix = subject_matrix 
        self.rel_matrix = rel_matrix 
        self.object_matrix = object_matrix

        self.loss = nn.BCELoss()

        ## checks
        if self.subject_matrix.shape[0] != self.rel_matrix.shape[0]:
            raise ValueError(f'Unexpected shape of subject_matrix or relation matrix. Expected dimension 0 to be same in subject_matrix and rel_matrix')

    def forward(self, trans_input):
        model_output = self.trans_model(**trans_input)
        pred_relations = mean_pooling(model_output, trans_input['attention_mask'])

        predicted_objects = self.follow(self, subject, pred_relations, subject_matrix, rel_matrix, object_matrix)

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
        subject_vectors = torch.sparse.mm(subject_matrix, subject) # (num_triplets, bs)

        relation = torch.transpose(relation, 0, 1)
        relation_vectors = torch.sparse.mm(rel_matrix, relation) # (num_triplets, bs)

        object_query = subject_vectors * relation_vectors

        predicted_objects = torch.sparse.mm(object_matrix, object_query) # (num_entities, bs)
        predicted_objects = torch.transpose(predicted_objects, 0, 1) # (bs, num_entities)

        return predicted_objects


    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    
    def shared_step(self, batch, dataname='train'):
        x, y = batch

        y_pred = self(x)
        loss = self.loss(y_pred, y)

        if dataname == 'train':
            on_step = True
        else:
            on_step = False
            
        with torch.no_grad():
            acc = calculate_accuracy(y_pred, y)
            self.log(f'{dataname}_acc', acc, on_step=on_step, on_epoch=True, prog_bar=True)
            
            if dataname != 'train':
                y_pred_norm = torch.softmax(y_pred, dim=-1)
                topk_acc = self.get_acc_topk(y_pred_norm, y).item()
                self.log(f'{dataname}_acc_top_{self.topk}',
                         topk_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log(f'{dataname}_loss', loss, on_step=on_step,
                 on_epoch=True, prog_bar=True)
        
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return 

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


