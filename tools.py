import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# reverse a dictionary, let the key become value and value become key
def reverse_dict(dict_src):
    dict_inversed = {}
    for key in dict_src.keys():
        dict_inversed[dict_src[key]] = key
    return dict_inversed
# Determine keywords for all categories, the keyword was selected by the coefficent(weight) of the word in
# the trained model
# Input:
#    vectorizer:  the vectorizer used to convert text to BOW
#    model:       trained supervised model
#    num:         Number of keywords you want to find
#    words_count: numpy array with dimension [1,size_of_vocabulary].
#                 The total number of times each word appears in the dataset
#    count_thres: Keywords must appeared more than this number
#    for_pos:     Whether to look for keywords for positive prediction 
# Output:
#    important_word: list of string
#
def show_word(vectorizer,model,num,words_count,count_thres = 5,for_pos = True):
    index_2_word_dict = reverse_dict(vectorizer.vocabulary_)
    # get positive numbers 
    if for_pos:
        coef = model.coef_.todense()
    else:
        coef = -model.coef_.todense()
    pos_indices = np.where(coef>0)[1]
    pos_word = coef[0,pos_indices]
    descending_sort_pos_words = np.argsort(-pos_word)
    important_word_index = []
    important_word = []
    count =0
    index_count = 0
    while count < num:
        #import pdb; pdb.set_trace()
        #coef[0,pos_indices[descending_sort_pos_words[0,0]]]
        tmp_index = pos_indices[descending_sort_pos_words[0,index_count]]
        if words_count[0,tmp_index] > count_thres:
            important_word_index.append(tmp_index)
            count+=1
        index_count += 1
    print(coef[0,tmp_index])
    for index in important_word_index:
        important_word.append(index_2_word_dict[index])
    return important_word
    
# Convert splited sentence to documents.
# Input:
#   doc_list:     List of list. each element in the list is a document, represented as list of all document's sentences
# Output:
#   new_doc_list: List of string. each element in the list is still a document, but the sentences were combined together
def back_to_doc(doc_list): 
    new_doc_list = []
    for doc in doc_list:
        new_doc_list.append('')
        for sent in doc:
            new_doc_list[-1]+=str(sent)
    return new_doc_list
    

def default_bow(new_text_total):
    Count_vectorizer =CountVectorizer(ngram_range=(1,1),min_df =5,stop_words='english',max_df = 0.8)
    Count_vec = Count_vectorizer.fit_transform(new_text_total)
    return Count_vec