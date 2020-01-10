import pickle
from sklearn.feature_extraction.text import CountVectorizer
import torch
import pdb
from torch import optim
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tokenization
import time
import re
import sklearn
from collections import defaultdict
import os
import numpy as np

# Custom pytorch dataset object that can pass text to bert's in the form of sentences
# input of init method:
#   Doc_list: list of list.
#   labels:   list of scale
class DocDataset(Dataset):
    def __init__(self, Doc_list, labels):
        #'Initialization'
        self.labels = labels
        self.Doc_list = Doc_list
    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.Doc_list)
    def __getitem__(self, index):
        #'Generates one sample of data'
        # Load data and get label
        X = self.Doc_list[index]
        y = self.labels[index]
        return X, y
 
# Load Doc_list (list of list) data and labels
# Output:
# training_generator:   Pytorch generator that return training samples
# validation_generator: Pytorch generator that return testing samples
# dummy_generator:      Generator that only return 5 samples, for fast debug the whole process
def load_data():
    src_dir = 'pu_data_ratio_Clarity2/'
    text_train = pickle.load(open(src_dir+'text_train_stratified.pkl','rb'))
    label_train = pickle.load(open(src_dir+'label_train_stratified.pkl','rb'))
    text_test = pickle.load(open(src_dir+'text_test_stratified.pkl','rb'))
    label_test = pickle.load(open(src_dir+'label_test_stratified.pkl','rb'))
    params_sent = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 6}
    params_sent_validation = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 6}
    training_set = DocDataset(text_train, label_train)
    training_generator = DataLoader(training_set, **params_sent)

    validation_set = DocDataset(text_test, label_test)
    validation_generator =DataLoader(validation_set, **params_sent_validation)
    dummy_set = DocDataset(text_train[:5], label_train[:5])
    dummy_generator = DataLoader(dummy_set, **params_sent)
    return training_generator,validation_generator,dummy_generator
# Convert splited sentence to documents. 

def load_raw_data(ratio=2,clarity=False,stratify = True):
    src_dir = 'pu_data_ratio' + str(ratio) + '/'
    if clarity:
        src_dir = 'pu_data_ratio_Clarity' + str(ratio) + '/'
    if stratify:
        text_train = pickle.load(open(src_dir+'text_train_stratified.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train_stratified.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test_stratified.pkl','rb'))
        label_test = pickle.load(open(src_dir+'label_test_stratified.pkl','rb'))
    else:
        text_train = pickle.load(open(src_dir+'text_train.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
    return text_train,label_train,text_test,label_test
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
            new_doc_list[-1]+=' ' + str(sent)
    return new_doc_list
    
def read_raw_data(ratio=2,clarity=False,stratify = True):
    src_dir = 'pu_data_ratio' + str(ratio) + '/'
    if clarity:
        src_dir = 'pu_data_ratio_Clarity' + str(ratio) + '/'
    if stratify:
        text_train = pickle.load(open(src_dir+'text_train_stratified.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train_stratified.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test_stratified.pkl','rb'))
        label_test = pickle.load(open(src_dir+'label_test_stratified.pkl','rb'))
    else:
        text_train = pickle.load(open(src_dir+'text_train.pkl','rb'))
        label_train = pickle.load(open(src_dir+'label_train.pkl','rb'))
        text_test = pickle.load(open(src_dir+'text_test.pkl','rb'))
        label_test = pickle.load(open(src_dir+'label_test.pkl','rb'))
    text_train = back_to_doc(text_train)
    text_test = back_to_doc(text_test)
    return text_train,label_train,text_test,label_test
    
# Concatenate short sentences into a sentence longer than 20 words 
# Input:
#   sent_list:          list of string, each string is a sentence.
# Output:
#   extended_sent_list: list of string.
def cancat_to_20(sent_list):
    extended_sent_list = []
    sent_index = 0
    while sent_index < len(sent_list):
        tmp_long_sent = ''
        tmp_word_num = 0
        while tmp_word_num < 20 and sent_index < len(sent_list):
            tmp_long_sent += sent_list[sent_index] +' '
            tmp_word_num += len(sent_list[sent_index].split())
            sent_index +=1
        extended_sent_list.append(tmp_long_sent)
    extended_sent_list.append(tmp_long_sent)
    return extended_sent_list
        
# Find the index of the misclassified sample
# Input:
#    y:           ground truth
#    y_hat:       predicted label
#    statis_dict: dict that has keys 'false_positive' and 'false_negative'
# Output:
#    statis_dict
def wrong_statis(y,y_hat,statis_dict):
    y = np.array(y)
    y_hat = np.array(y_hat)
    diff = y - y_hat
    false_positive_indices = np.where(diff == -1)[0]
    false_negative_indices = np.where(diff ==  1)[0]
    if np.sum(y_hat):
        statis_dict['false_positive'].append(false_positive_indices)
        statis_dict['false_negative'].append(false_negative_indices)
    return statis_dict
# Find all models in the folder and return one by one
# Input:
#   src_dir: path to model files
# Output:
#   word_model: pytorch state_dict
#   sent_model: pytorch state_dict
def dir_model_iter(src_dir):
    index_list = []
    for file in os.listdir(src_dir):
        if 'save_word' in file:
            cur_index = re.search('\d+',file).group()
            index_list.append(int(cur_index))
    index_list.sort()
    for index in index_list:
        word_model = 'save_word_' + str(index) + '.bin'
        sent_model = 'save_sent_' + str(index) + '.bin'
        yield word_model,sent_model
        
# Input:
#   vocab_file: vocab_file
#   uncased:    whether cased or uncased
# Output:
#   tf_tokenizer: Bert's tokenizer
def _load_tf_tokenizer(vocab_file=None,uncased=True):
    if vocab_file is None:
        vocab_file ='/gpfs/qlong/home/tzzhang/mimicIII/nn_code/biobert_pretrain_output_all_notes_150000/vocab.txt'
    tf_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=uncased)
    return tf_tokenizer

# Load sentences from a list of sentence
# Input:
#   doc_text:    list of sentence
#   batch_size:  scalar
#   batch_count: number of current batch
#   max_doc_len: maximal number of sentences
# Output:
#   batch_sent: list of string
def batch_sent_loader(doc_text,batch_size,batch_count,max_doc_len=400):
    start = batch_size * batch_count 
    upper_bound = min(len(doc_text),max_doc_len-2) # Leave space for two special tokens
    end = min(upper_bound, batch_size * (batch_count+1))
    batch_sent = doc_text[start:end]
    return batch_sent

def word_tokenize(text,max_seq_length,tokenizer=None):
    #print(text)
    if tokenizer is None:
        tokenizer = _load_tf_tokenizer()
    text = text.replace('\n',' ')
    raw_tokens = tokenizer.tokenize(text)
    if len(raw_tokens) > max_seq_length - 2:
        raw_tokens = raw_tokens[0:(max_seq_length - 2)]
    tokens = []
    tokens.append("[CLS]")
    for token in raw_tokens:
        tokens.append(token)
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    return input_ids,input_mask
 
# Handy class for restore all hyper-parameters 
class hpara:
    def __init__(self):
        self.word_lr = 0.00005
        self.sent_lr = 0.0002
        self.max_epoch = 60
        self.batch_size = 256
        self.accumulation_steps = 40
        self.max_sent_len = 64
        self.max_doc_len = 400
        self.ratio = 2
        self.word_layers = 6
        self.sent_layers = 3
        self.hidden_size = 768
        self.use_angular = False
        self.use_PU_Bert = True
            
# Implementation of model training and testing
# Input:
#   hpara1: hpara class
#   model_word: pytorch model
#   model_sent: pytorch model
#   save_dir:   path to save trained model and log
#   training_generator:  generator to load training data
#   validation_generator: generator to load testing data
#   tokenizer:  Bert's tokenizer
def model_train_and_test(hpara1,model_word,model_sent,save_dir,\
                        training_generator,validation_generator, \
                        tokenizer,do_train=True,do_test=True):
    # Set up hyper parameters
    criterion = torch.nn.NLLLoss()
    log_file = open(save_dir+'log','a')
    max_epoch = hpara1.max_epoch
    log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},time={:.3f}\n'
    test_log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},auc={:.3f},time={:.3f}\n'
    batch_size = hpara1.batch_size
    accumulation_steps = hpara1.accumulation_steps
    max_sent_len = hpara1.max_sent_len
    max_doc_len = hpara1.max_doc_len
    optimizer = optim.Adamax
    word_optimizer = optimizer(model_word.parameters(), lr=hpara1.word_lr)
    sent_optimizer = optimizer(model_sent.parameters(), lr=hpara1.sent_lr)
    para_dict = {}
    hpara_list = []
    cls_weight = torch.tensor(np.load('cls_weight.npy')).cuda()
    sep_weight = torch.tensor(np.load('sep_weight.npy')).cuda()
    # do actual training and testing
    
    for cur_epoch in range(0,max_epoch):
        log_file = open(save_dir+'log','a')
        if do_train:
            start = time.time()
            model_sent.train()
            model_word.train()
            correct = sum_loss =total_num = 0
            for doc_count,(doc,label) in enumerate(training_generator):
                total_num += len(label)
                if doc_count%1000 ==0:
                    print(doc_count)
                label = label.cuda()
                batch_count=0
                sent_num=0
                end_ind=0
                input_tensors = torch.zeros([1,max_doc_len,hpara1.hidden_size]).cuda()
                # Add cls in sent level
                input_tensors[0,0] = cls_weight
                while end_ind < len(doc) and end_ind < max_doc_len-1:
                    batch_sent = batch_sent_loader(doc,batch_size,batch_count,max_doc_len=max_doc_len)
                    cur_batch_size = len(batch_sent)
                    sent_num += cur_batch_size
                    input_ids = torch.zeros(cur_batch_size,max_sent_len).long().cuda()
                    input_mask = torch.ones(cur_batch_size,max_sent_len).long().cuda()
                    for i in range(len(batch_sent)):
                        tmp_ids,tmp_mask = word_tokenize(batch_sent[i][0],max_sent_len,tokenizer)
                        input_ids[i,:] = torch.tensor(tmp_ids)
                        input_mask[i,:] = torch.tensor(tmp_mask)
                    #pdb.set_trace()
                    _,tmp_input_tensors,word_att_output = model_word(input_ids,attention_mask=input_mask)
                    start_ind = batch_count*batch_size + 1 # because the cls was added in 0-th raw 
                    end_ind = start_ind + cur_batch_size
                    input_tensors[0,start_ind:end_ind] = tmp_input_tensors
                    batch_count +=1
                # -----------Add sep in sent matrix----------

                input_tensors[0,end_ind] = sep_weight
                sent_mask = [1]*(end_ind+1)
                while len(sent_mask)<max_doc_len:
                    sent_mask.append(0)
                sent_mask = torch.tensor(sent_mask).unsqueeze(0).cuda()
                if hpara1.use_angular:
                    loss,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                else:
                    _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                    loss = criterion(proba, label)
                #pdb.set_trace()
                sum_loss += loss.item() * len(label)
                _,predicted = torch.max(proba,1)
                correct += (predicted == label).sum()
                loss = loss / accumulation_steps                # Normalize our loss (if averaged)
                loss.backward()                                 # Backward pass
                if (doc_count+1) % accumulation_steps == 0:             # Wait for several backward steps
                    sent_optimizer.step()                           # Now we can do an optimizer step
                    word_optimizer.step()
                    word_optimizer.zero_grad()                           # Reset gradients tensors
                    sent_optimizer.zero_grad()
            torch.save(model_word.state_dict(),save_dir+'/save_word_'+str(cur_epoch)+'.bin')
            torch.save(model_sent.state_dict(),save_dir+'/save_sent_'+str(cur_epoch)+'.bin')
            accu = correct.item() / total_num
            to_print = log.format(cur_epoch,max_epoch,sum_loss,accu,time.time() - start)
            print(to_print)
            log_file.writelines(to_print)
        if do_test:
            start = time.time()
            model_sent.eval()
            model_word.eval()
            pred_list = []
            y_list = []
            y_hat = []
            correct = sum_loss =total_num = 0
            for doc_count,(doc,label) in enumerate(validation_generator):
                total_num += len(label)
                label = label.cuda()
                batch_count=0
                sent_num = 0
                end_ind = 0
                input_tensors = torch.zeros([1,max_doc_len,hpara1.hidden_size]).cuda()
                # Add cls in sent level
                input_tensors[0,0] = cls_weight
                while end_ind < len(doc) and end_ind < max_doc_len-1:
                    batch_sent =batch_sent_loader(doc,batch_size,batch_count,max_doc_len=max_doc_len)
                    cur_batch_size = len(batch_sent)
                    sent_num += cur_batch_size
                    input_ids = torch.zeros(cur_batch_size,max_sent_len).long().cuda()
                    input_mask = torch.ones(cur_batch_size,max_sent_len).long().cuda()
                    for i in range(len(batch_sent)):
                        tmp_ids,tmp_mask= word_tokenize(batch_sent[i][0],max_sent_len,tokenizer)
                        input_ids[i,:] = torch.tensor(tmp_ids)
                        input_mask[i,:] = torch.tensor(tmp_mask)
                    _,tmp_input_tensors,word_att_output = model_word(input_ids,attention_mask=input_mask)
                    start_ind = batch_count*batch_size+1
                    end_ind = start_ind + cur_batch_size
                    input_tensors[0,start_ind:end_ind] = tmp_input_tensors
                    batch_count +=1
                # -----------Add sep in sent matrix----------
                input_tensors[0,end_ind] = sep_weight
                sent_mask = [1]*(end_ind+1)
                while len(sent_mask)<max_doc_len:
                    sent_mask.append(0)
                sent_mask = torch.tensor(sent_mask).unsqueeze(0).cuda()
                if hpara1.use_angular:
                    loss,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                else:
                    _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
                    loss = criterion(proba, label)            
                pos_socre = np.exp(proba.cpu().detach().numpy())[0,1]
                y_hat.append(pos_socre)
                loss = criterion(proba, label)
                _,predicted = torch.max(proba,1)
                #for making confusion matrix
                pred_list.append(predicted.item())
                y_list.append(label.item())

                sum_loss += loss.item() * len(label)
                correct += (predicted == label).sum()
            accu = correct.item() / total_num
            roc_score = sklearn.metrics.roc_auc_score(y_list, y_hat)
            to_print=test_log.format(cur_epoch,max_epoch,sum_loss,accu,roc_score,time.time() - start)
            to_print = 'Test ' + to_print
            print(to_print)
            print(confusion_matrix(y_list,pred_list))
            log_file.writelines(to_print)
            log_file.close()
            
def output_att_scores(hpara1,model_word,model_sent,save_dir,\
                        training_generator,validation_generator, \
                        use_angular=True):               
    # Set up hyper parameters
    criterion = torch.nn.NLLLoss()
    log_file = open(save_dir+'log','a')
    max_epoch = hpara1.max_epoch
    log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},time={:.3f}\n'
    test_log = 'Iter {}/{}, Loss={:.3f},accu={:.3f},auc={:.3f},time={:.3f}\n'
    batch_size = hpara1.batch_size
    accumulation_steps = hpara1.accumulation_steps
    max_sent_len = hpara1.max_sent_len
    max_doc_len = hpara1.max_doc_len
    #progress_bar = tqdm(enumerate(training_generator))
    optimizer = optim.Adamax
    word_optimizer = optimizer(model_word.parameters(), lr=hpara1.word_lr)
    sent_optimizer = optimizer(model_sent.parameters(), lr=hpara1.sent_lr)
    para_dict = {}
    hpara_list = []
    tokenizer = _load_tf_tokenizer(vocab_file = '/gpfs/qlong/home/tzzhang/nlp_test/bert/mimic_based_complete_model/vocab.txt')
    cls_weight = torch.tensor(np.load('cls_weight.npy')).cuda()
    sep_weight = torch.tensor(np.load('sep_weight.npy')).cuda()
    
    model_sent.eval()
    model_word.eval()
    pred_list = []
    y_list = []
    y_hat = []
    correct = sum_loss =total_num = 0
    for doc_count,(doc,label) in enumerate(validation_generator):
        word_att_list = []
        total_num += len(label)
        label = label.cuda()
        batch_count=0
        sent_num = 0
        end_ind = 0
        input_tensors = torch.zeros([1,max_doc_len,hpara1.hidden_size]).cuda()
        # Add cls in sent level
        input_tensors[0,0] = cls_weight
        while end_ind < len(doc) and end_ind < max_doc_len:
            batch_sent =batch_sent_loader(doc,batch_size,batch_count,max_doc_len=max_doc_len)
            cur_batch_size = len(batch_sent)
            sent_num += cur_batch_size
            input_ids = torch.zeros(cur_batch_size,max_sent_len).long().cuda()
            input_mask = torch.ones(cur_batch_size,max_sent_len).long().cuda()
            for i in range(len(batch_sent)):
                tmp_ids,tmp_mask= word_tokenize(batch_sent[i][0],max_sent_len,tokenizer)
                input_ids[i,:] = torch.tensor(tmp_ids)
                input_mask[i,:] = torch.tensor(tmp_mask)
            _,tmp_input_tensors,word_att_output = model_word(input_ids,attention_mask=input_mask)
            word_att_list.append(word_att_output)
            start_ind = batch_count*batch_size+1
            end_ind = start_ind + cur_batch_size
            input_tensors[0,start_ind:end_ind] = tmp_input_tensors
            batch_count +=1
        # -----------Add sep in sent matrix----------
        if end_ind<max_doc_len:
            input_tensors[0,end_ind] = sep_weight
        else:
            end_ind = max_doc_len-1
            input_tensors[0,end_ind] = sep_weight
        sent_mask = [1]*(end_ind+1)
        while len(sent_mask)<max_doc_len:
            sent_mask.append(0)
        sent_mask = torch.tensor(sent_mask).unsqueeze(0).cuda()
        if hpara1.use_angular:
            loss,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
        else:
            _,proba,sent_att_output = model_sent(input_tensors,label,attention_mask=sent_mask)
            loss = criterion(proba, label)
        yield word_att_list,sent_att_output,doc,proba
            
                   

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
    
def default_bow(new_text_total):
    Count_vectorizer =CountVectorizer(ngram_range=(1,1),min_df =5,stop_words='english',max_df = 0.8)
    Count_vec = Count_vectorizer.fit_transform(new_text_total)
    return Count_vec