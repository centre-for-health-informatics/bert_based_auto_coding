## automatic pressure ulcer detection
Developed a pressure ulcer(PU) automatic detection algorithm based on the discharge summary in mimic-III. The advanced and popular bert model is adopted. Document embedding was obtained by another bert that take sentence embedding as the input, it was a HAN(hierarchical attention networks). The classification of the document embedding is used to predict the pressure ulcer.

### Dependencies
+ Python 3.5.2 
+ torch 1.3.1
+ pytorch_transformers 1.0
+ numpy 1.16.1
+ sklearn 0.21.3
+ ClarityNLP

### Pre-processing
#### 1. Define positive samples
All discharge summaries that has one of ICD code
'70709','70708','70707','70706','70705','70704','70703','70702','70701','70700','70725','70724','70723','70722','70721','70720' are considered as positive samples。

#### 2. Ratio of positive : negative sample
Only 3.5% of the discharge summary in MIMIC III data have PU according to the ICD code. Taking the entire MIMIC III data as training data will cause a very serious bias. However, if make the positive and negative samples in the training set to be 1: 1,  this does not reflect the prevalence of PU at all, and will cause a large number of false positives in practice. In the end, I took a ratio positive :negative = 1: 2. One to three and one to ten have also been tested, but the performance is not as good as one to two.

#### Sentence tokenize
Clarity NLP has a specific sentence tokenizer for MIMIC III data, it indeed works better than the default sentence tokenizer in spacy according to my experience. A more detailed explanation can be found on clarity NLP's website:
[ClarityNLPSentenceTokenizer](https://claritynlp.readthedocs.io/en/latest/developer_guide/algorithms/sentence_tokenization.html)

After downloading ClarityNLP, the two line in pre-processing_with_clarity_seg should be modified.
```
sys.path.append('path_to_ClarityNLP/ClarityNLP/nlp')
sys.path.append('path_to_ClarityNLP/ClarityNLP/')
```

### Model
[Model_Illustration](./illustration_of_model.png)

I merged HAN (hierarchical attention networks) and bert. It's like stack another bert model on the original bert, which only has sentence-level attention.

The training code can be found in utils.py, which has detailed comments.

For the model constructor, I used the bert class in pytorch_transformers. I write a custom model class, which is basically a bert model without the first embedding layer (the layer convert word's one hot representation into a dense vector), as the document level bert. This custom class can be found in modeling_bert.py  


