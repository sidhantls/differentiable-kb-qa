import os
import re
import torch

def load_triplets(path: str, keep_only_entities: dict = {}):   
    """
    Given path of a text file of knowledge graph, load head, rel, tail relations

    Follows the format of MetaQA, kg.txt
    
    Inputs:
        - path to file 
        - keep_only_entities - contains dict of those entities that we want to keep in KB. This is only useful 
            for aligning the large metaqa with the new updated metaqa dataset. Keep this empty for all other usecases

    Returns:
        - triplets: list of tuples (headid, relid, tailid)
        - entity_to_idx- entity name to idx dict
        - rel_to_idx - relation name to idx dict
        - idx_to_entity
        - idx_to_rel
    """
    entity_to_idx, rel_to_idx = {}, {}
    ent_idx, rel_idx = -1, -1

    triplets = []

    if not os.path.exists(path):
        print(f'Path {path} does not exist')

    with open(path, 'r', encoding="utf8") as f: 
        for line in f:
            line = line.strip()
            head, rel, tail = line.split('|')
            
            # only for special case
            if keep_only_entities and (head not in keep_only_entities or tail not in keep_only_entities):
                continue
                
            # add entity idx to mapping
            for ent in [head, tail]: 
                if ent not in entity_to_idx:
                    ent_idx += 1
                    entity_to_idx[ent] = ent_idx
                
            
            # add relation idx to mapping
            if rel not in rel_to_idx: 
                rel_idx += 1
                rel_to_idx[rel] = rel_idx            
            
            # create triplet (head_idx, relation_idx, tail_idx)
            curr_head_idx, curr_rel_idx, curr_tail_idx = entity_to_idx[head], rel_to_idx[rel], entity_to_idx[tail]
            
            
            triplets.append((curr_head_idx, curr_rel_idx, curr_tail_idx))
            
    print('num entities:', len(entity_to_idx))
    print('num relations:', len(rel_to_idx))
    print('num triplets ', len(triplets))

    idx_to_entity = {idx: ent for ent, idx in entity_to_idx.items()}
    idx_to_rel = {idx: rel for rel, idx in rel_to_idx.items()}

    return triplets, entity_to_idx, rel_to_idx, idx_to_entity, idx_to_rel



def load_triplets_metaqa(path: str, not_inverse_fields = ['release_year', 'in_language', 'has_genre', 'has_tags']):   
    """
    Given path of a text file of knowledge graph, load head, rel, tail relations

    Follows the format of MetaQA, kg.txt
    
    Inputs:
        - path to file 
        - keep_only_entities - contains dict of those entities that we want to keep in KB. This is only useful 
            for aligning the large metaqa with the new updated metaqa dataset. Keep this empty for all other usecases

    Returns:
        - triplets: list of tuples (headid, relid, tailid)
        - entity_to_idx- entity name to idx dict
        - rel_to_idx - relation name to idx dict
        - idx_to_entity
        - idx_to_rel
    """
    entity_to_idx, rel_to_idx = {}, {}
    ent_idx, rel_idx = -1, -1

    triplets = []

    if not os.path.exists(path):
        print(f'Path {path} does not exist')

    with open(path, 'r', encoding="utf8") as f: 
        for line in f:
            line = line.strip()
            head, rel, tail = line.split('|')

            # add entity idx to mapping
            for ent in [head, tail]: 
                if ent not in entity_to_idx:
                    ent_idx += 1
                    entity_to_idx[ent] = ent_idx
                
            
            # add relation idx to mapping
            if rel not in rel_to_idx: 
                rel_idx += 1
                rel_to_idx[rel] = rel_idx            
            
            # create triplet (head_idx, relation_idx, tail_idx)
            curr_head_idx, curr_rel_idx, curr_tail_idx = entity_to_idx[head], rel_to_idx[rel], entity_to_idx[tail]
            
            triplets.append((head, rel, tail))
            
            # add inverse relationship
            # according to benchmark - https://github.com/google-research/language/blob/master/language/emql/preprocess/metaqa_preprocess.py#L63
             
            if rel not in not_inverse_fields:
                rel = rel + '-inv'
                triplets.append((tail, rel, head))
                
                if rel not in rel_to_idx: 
                    rel_idx += 1
                    rel_to_idx[rel] = rel_idx
                
            
            
    print('num entities:', len(entity_to_idx))
    print('num relations:', len(rel_to_idx))
    print('num triplets ', len(triplets))

    idx_to_entity = {idx: ent for ent, idx in entity_to_idx.items()}
    idx_to_rel = {idx: rel for rel, idx in rel_to_idx.items()}

    return triplets, entity_to_idx, rel_to_idx, idx_to_entity, idx_to_rel



def load_qa_pairs(path):
    """
    Loads qa pairs from text file. Text file follows the format in MetaQA

    Returns list of tuples of 3 items:
        - [(question string, answer entities: tuple, subject entity: tuple)]
    """

    qa_pairs = []
    with open(path, encoding="utf8") as f:
        for line_nb, line in enumerate(f):
            question, answer = line.rstrip().split('\t')
            
            subject_entities = re.findall('\[(.*?)\]', question)
            
            if subject_entities and answer: 
                question = question.replace('[', ' ').replace(']', ' ')
                answers = answer.split('|')
                answers = tuple(map(lambda x: x.strip(), answers))
                
                qa_pairs.append((question, answers, tuple(subject_entities)))
                
            else:
                warnings.warn(f'No entity found (either question in [] or in answer) in line {line_nb}')
                
        print('Num qa pairs loaded:', len(qa_pairs))


    return qa_pairs


def santity_check(qa_pairs, entity_to_idx):
    """
    Verify qa pairs are in entities in kb
    """

    missing = []
    for question, answer_entities, subject_entities in qa_pairs: 
        for ent in answer_entities: 
            if ent not in entity_to_idx:
                missing.append(ent)
            
        for ent in subject_entities:
            if ent not in entity_to_idx:
                missing.append(ent)
    
    print(f'number of entities missing {len(missing)}: {missing[:3]}')



def create_sparse_tensor(ent_to_idx, triplets, name='head'):
    """
    Given triplets and ent_to_idx/or rel_to_idx, create the sparse kg tensor

    Arguments:
        - ent_to_idx: dict mapping entity to idx 
        - triplets: list of tuples containing (headidx, relationidx, tailidx)
        - name: which entity to get tensor of - either head/tail entity, or  relation type.
                based on the name, we index the triplet create vector

    
    """

    # create sparse tensor
    if name == 'head':
        cols = [entity_idx for (entity_idx, _, _) in triplets]
    elif name == 'relation': 
        cols = [rel_idx for (_, rel_idx, _) in triplets]
    elif name == 'tail': 
        cols = [entity_idx for (_, _, entity_idx) in triplets] 
        
    else:
        raise NotImplementedError('node name should be either head, tail or relation')
        
    rows = [idx for idx in range(len(triplets))]
    indices = [rows, cols]
    values = torch.FloatTensor([1.]*len(triplets)) 
     
    shape = (len(triplets), len(ent_to_idx))
    
    M = torch.sparse_coo_tensor(indices, values, shape)
    del cols; del rows; del indices; del values

    return M
    
    
def create_differentiable_kg(triplets, entity_to_idx, rel_to_idx):
    """
    Create the 3 matrices for differentiable kg - Ms, Mo, Mr
    """
    triplet_ids = [(entity_to_idx[row[0]], rel_to_idx[row[1]], entity_to_idx[row[2]]) for row in triplets]
    
    Ms = create_sparse_tensor(entity_to_idx, triplet_ids, name='head')
    Mr = create_sparse_tensor(rel_to_idx, triplet_ids, name='relation')
    Mo = create_sparse_tensor(entity_to_idx, triplet_ids, name='tail')
    
    return Ms, Mr, Mo


