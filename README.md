# QA on Differentiable Knowledge Bases- Reified KB
This is an unofficial adaption of the work presented in Scalable Neural Methods for Reasoning With a Symbolic Knowledge Base- [Paper](https://arxiv.org/abs/2002.06115). Utilizes a transformer based encoder, similar to the work [here](https://arxiv.org/abs/2109.05808v1), as opposed to word2vec which is how it was implemented in the Symbolic KB paper

The purpose of this implementation is to provide insights into the implementation level information of these papers and the ideas asscociated with it - such as enabling a differentiable KG through the "follow" operation. 


## Implemenation
Implements a QA model to retrieve questions from a KB. This consists of an encoder, MiniLM-6/MiniLM-12, which encodes a question. And then, it performs the differentiable query operation on the knowledge base as mentioned in the reified KB paper above. This encode archicture is different from the paper in that it's not based on a word2vec model, but a tranformer. We fine-tune a MiniLM-6 Sentence Transformer - primarily because of the memory and compute efficiency. Can replace this with any other model from the hugginface library 


### How to run: 
* Refer to kgqa/models 

### Benchmarks
Bencharked on the [MetaQA](https://github.com/yuyuz/MetaQA) dataset, similar to the other symbolic kb paper linked above

| MetaQA      | Hit @k =1  |
| ----------- | ----------- |
| 1-hop       | 0.935       |
| 2-hop       | 0.769       |
| 3-hop       | 0.803       |

One expectation was that 3 hop performance to be worse than 2-hop, which was the case in reified kb paper as well. The reified KB paper does report slightly higher 1-hop and 2-hop performance, but this 3-hop outperforms it. 

### User Experimentation

Users are free to use this codebase as a starting point to implement improvements in this space of question answering with reified differential KBs

* Update encoder:
    * Utilize different transformer model - Pass in any model from Hugginface library 
    * To change encoder to word2vec, update the torch dataset (train_model_metaqa) and forward pass in net (models_nhop)
* Implement updated architectures- for eg. to not require subject_id as part of model input - perform entity recongition and train end to end


### To add
    * To add training and inference on custom datasets
    * Method to integrate 1hop, 2hop, 3hop one training procedure


