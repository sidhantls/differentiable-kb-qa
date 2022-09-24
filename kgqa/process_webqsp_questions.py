import json 
import os 
from pathlib import Path 

def write_qa_triplets(path, write_path, write_path2):
    with open(path, encoding='utf-8') as f: 
        data = json.load(f)

    with open(write_path, 'w', encoding='utf-8') as f:
        f.write('')
       
    with open(write_path2, 'w', encoding='utf-8') as f:
        f.write('')

    for question in data['Questions']:
        question_string = question['RawQuestion']
        relation_path = question['Parses'][0]['InferentialChain']
        
        answer_entities = [row['EntityName'] for row in question['Parses'][0]['Answers']]
        answer_entity_ids = [row['AnswerArgument'] for row in question['Parses'][0]['Answers']]
        
        subject_entities = question['Parses'][0]['TopicEntityName']
        subject_entity_id = question['Parses'][0]['TopicEntityMid']

        # import pdb; pdb.set_trace()

        with open(write_path, 'a', encoding='utf-8') as f:
            if isinstance(question_string, str) and isinstance(relation_path, list) and relation_path and relation_path[0] and isinstance(answer_entities, list) and answer_entities and answer_entities[0] and isinstance(subject_entities, str):
                # print(answer_entities)
                f.write(f"{question_string}\t{subject_entities}\t{'|'.join(relation_path)}\t{'|'.join(answer_entities)}\n")

        with open(write_path2, 'a', encoding='utf-8') as f:
            if isinstance(question_string, str) and isinstance(relation_path, list) and relation_path and relation_path[0] and isinstance(answer_entity_ids, list) and answer_entity_ids and answer_entity_ids[0] and isinstance(subject_entity_id, str):
                # print(answer_entities)
                f.write(f"{question_string}\t{subject_entity_id}\t{'|'.join(relation_path)}\t{'|'.join(answer_entity_ids)}\n")
                
root_dir = Path(os.getcwd())


if not os.path.exists(root_dir/'datasets/WebQSP/data_processed'):
    os.mkdir(root_dir/'datasets/WebQSP/data_processed')


path = root_dir/'datasets/WebQSP/data/WebQSP.train.json'
write_path = root_dir/'datasets/WebQSP/data_processed/WebQSP_train.txt'

# write qa pairs, but answer and subject entity ids, not entity names
write_path2 = root_dir/'datasets/WebQSP/data_processed/WebQSP_train_entids.txt' 

write_qa_triplets(path, write_path, write_path2)

path = root_dir/'datasets/WebQSP/data/WebQSP.test.json'
write_path = root_dir/'datasets/WebQSP/data_processed/WebQSP_test.txt'
write_path2 = root_dir/'datasets/WebQSP/data_processed/WebQSP_test_entids.txt'

write_qa_triplets(path, write_path, write_path2)


    

        


