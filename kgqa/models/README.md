## Models 

### MetaQA Benchark:
Question encoder is a MiniLM-6 model. Deviates from the paper which uses W2V. 

#### How to run: 
* cd into the kgqa/models
* `python train_model_metaqa.py --num_hops=1`. num hops can be 1,2,3 
* Models: `models_nhop.py` contains model and differential kb follow

| MetaQA      | Hit @k =1  |
| ----------- | ----------- |
| 1-hop       | 0.935       |
| 2-hop       | 0.769       |
| 3-hop       | 0.803       |



