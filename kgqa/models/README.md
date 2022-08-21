## Models 

### MetaQA Benchark:
Question encoder is a MiniLM-6 model. Deviates from the paper which uses W2V. 

#### How to run: 
* cd into the kgqa/models
* `python train_model_metaqa.py --num_hops=1`. num hops can be 1,2,3 
* View training/eval metrics `tensorboard --logdir=kb_logs`
* Models: `models_nhop.py` contains model and differential kb follow


