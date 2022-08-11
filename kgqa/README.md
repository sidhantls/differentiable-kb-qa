# QA on Differentiable Knowledge Bases- Reified KB
This is an unofficial adaption of the work presented in Scalable Neural Methods for Reasoning With a Symbolic Knowledge Base- [Paper](https://arxiv.org/abs/2002.06115). Utilizes a transformer based encoder, similar to the work [here](https://arxiv.org/abs/2109.05808v1), as opposed to word2vec which is how it was implemented in the Symbolic KB paper

The purpose of this implementation is to provide insights into the implementation level information of these papers and the ideas asscociated with it - such as enabling a differentiable KG through the "follow" operation. 


## Implemenation
Implements a QA model to retrieve questions from a KB. This consists of an encoder, MiniLM-6/MiniLM-12, which encodes a question. And then, it performs the differentiable query operation on the knowledge base as mentioned in the reified KB paper above. This encode archicture is different from the paper in that it's not based on a word2vec model, but a tranformer. We fine-tune a MiniLM-6 Sentence Transformer - primarily because of the memory and compute efficiency. Can replace this with any other model from the hugginface library 


### How to run
Refer to kgqa/models 





