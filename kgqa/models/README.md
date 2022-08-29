## Models 

### MetaQA Benchark:
Question encoder is a MiniLM-6 model.

#### Data 
Download the [MetaQA](https://github.com/yuyuz/MetaQA) dataset into `kgqa/datasets/MetaQA`

```
kgqa
├── datasets
  ├── MetaQA
    ├── 1hop
    ├── 2hop
    ├── 3hop
    ├── kb.txt
```


#### How to run: 
* Get the metaqa dataset
* cd into the kgqa/models
* `python train_model_metaqa.py --num_hops=1`. num hops can be 1,2,3 
* View training/eval metrics `tensorboard --logdir=kb_logs`
* Models: `models_nhop.py` contains model and differential kb follow


