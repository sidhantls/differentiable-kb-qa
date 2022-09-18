
from torch.utils.data import Dataset, DataLoader
import torch

class QADataset(Dataset):
    def __init__(self, qa_pairs, q_tokens, entity_to_idx):
        self.qa_pairs = qa_pairs
        self.entity_to_idx = entity_to_idx
        self.q_tokens = q_tokens

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        token_sample = {key: self.q_tokens[key][idx, :] for key in self.q_tokens}
        
        head_entities = self.qa_pairs[idx][2]
        head_entities = [self.entity_to_idx[entity] for entity in head_entities]
                
        head_entities = self.create_onehot_entity_vector(head_entities)
        
        tail_entities = self.qa_pairs[idx][1]
        tail_entities = [self.entity_to_idx[entity] for entity in tail_entities]
        
        tail_entities = self.create_onehot_entity_vector(tail_entities)
        
        return token_sample, head_entities, tail_entities
    
    
    def create_onehot_entity_vector(self, entities):
        """
        Inputs: entities: list of ints 
        
        """
        num_entities = len(self.entity_to_idx)
        
        entity_tensor = torch.zeros(num_entities)
        entity_tensor[entities] = 1
        
        return entity_tensor

class QADataset2(Dataset):
    def __init__(self, qa_pairs, q_tokens, entity_to_idx):
        self.qa_pairs = qa_pairs
        self.entity_to_idx = entity_to_idx
        self.q_tokens = q_tokens

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        token_sample = {key: self.q_tokens[key][idx, :] for key in self.q_tokens}
        
        head_entities = self.qa_pairs[idx][2]
        head_entities = [self.entity_to_idx[entity] for entity in head_entities]
                
        head_entities = self.create_onehot_entity_vector(head_entities)
        
        tail_entities = self.qa_pairs[idx][1]
        tail_entities = [self.entity_to_idx[entity] for entity in tail_entities]
        
        tail_entities = self.create_onehot_entity_vector(tail_entities)

        num_hops = self.qa_pairs[idx][3]-1
        
        return token_sample, head_entities, tail_entities, num_hops
    
    
    def create_onehot_entity_vector(self, entities):
        """
        Inputs: entities: list of ints 
        
        """
        num_entities = len(self.entity_to_idx)
        
        entity_tensor = torch.zeros(num_entities)
        entity_tensor[entities] = 1
        
        return entity_tensor