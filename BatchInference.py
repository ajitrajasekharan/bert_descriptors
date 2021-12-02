import torch
import subprocess
#from pytorch_transformers import *
from transformers import *
import pdb
import operator
from collections import OrderedDict
import numpy as np
import argparse
import sys
import traceback
import string
import common as utils
import config_utils as cf
import requests

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


DEFAULT_TOP_K = 20
DEFAULT_MODEL_PATH='./'
DEFAULT_LABELS_PATH='./labels.txt'
DEFAULT_TO_LOWER=False
DESC_FILE="./common_descs.txt"
SPECIFIC_TAG=":__entity__"

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

def read_descs(file_name):
    ret_dict = {}
    with open(file_name) as fp:
        line = fp.readline().rstrip("\n")
        if (len(line) >= 1):
            ret_dict[line] = 1
        while line:
            line = fp.readline().rstrip("\n")
            if (len(line) >= 1):
                ret_dict[line] = 1
    return ret_dict

def read_vocab(file_name):
    l_vocab_dict = {}
    o_vocab_dict = {}
    with open(file_name) as fp:
        for line in fp:
            line = line.rstrip('\n')
            if (len(line) > 0):
                l_vocab_dict[line.lower()] = line   #If there are multiple cased versions they will be collapsed into one. which is okay since we have the original saved. This is only used
                                                    #when a word is not found in its pristine form in the original list.
                o_vocab_dict[line] = line
    print("Read vocab file:",len(o_vocab_dict))
    return o_vocab_dict,l_vocab_dict

def consolidate_labels(existing_node,new_labels,new_counts):
    """Consolidates all the labels and counts for terms ignoring casing

    For instance, egfr may not have an entity label associated with it
    but eGFR and EGFR may have. So if input is egfr, then this function ensures
    the combined entities set fo eGFR and EGFR is made so as to return that union
    for egfr
    """
    new_dict = {}
    existing_labels_arr = existing_node["label"].split('/')
    existing_counts_arr = existing_node["counts"].split('/')
    new_labels_arr = new_labels.split('/')
    new_counts_arr = new_counts.split('/')
    assert(len(existing_labels_arr) == len(existing_counts_arr))
    assert(len(new_labels_arr) == len(new_counts_arr))
    for i in range(len(existing_labels_arr)):
        new_dict[existing_labels_arr[i]] = int(existing_counts_arr[i])
    for i in range(len(new_labels_arr)):
        if (new_labels_arr[i] in new_dict):
            new_dict[new_labels_arr[i]] += int(new_counts_arr[i])
        else:
            new_dict[new_labels_arr[i]] = int(new_counts_arr[i])
    sorted_d = OrderedDict(sorted(new_dict.items(), key=lambda kv: kv[1], reverse=True))
    ret_labels_str = ""
    ret_counts_str = ""
    count = 0
    for key in sorted_d:
        if (count == 0):
            ret_labels_str = key
            ret_counts_str = str(sorted_d[key])
        else:
            ret_labels_str += '/' +  key
            ret_counts_str += '/' +  str(sorted_d[key])
        count += 1
    return {"label":ret_labels_str,"counts":ret_counts_str}


def read_labels(labels_file):
    terms_dict = OrderedDict()
    lc_terms_dict = OrderedDict()
    with open(labels_file,encoding="utf-8") as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) == 3):
                terms_dict[term[2]] = {"label":term[0],"counts":term[1]}
                lc_term = term[2].lower()
                if (lc_term in lc_terms_dict):
                     lc_terms_dict[lc_term] = consolidate_labels(lc_terms_dict[lc_term],term[0],term[1])
                else:
                     lc_terms_dict[lc_term] = {"label":term[0],"counts":term[1]}
                count += 1
            else:
                print("Invalid line:",term)
                assert(0)
    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict,lc_terms_dict


class BatchInference:
    def __init__(self, path,to_lower,patched,topk,abbrev,tokmod,vocab_path,labels_file):
        print("Model path:",path,"lower casing set to:",to_lower," is patched ", patched)
        self.path = path
        self.labels_dict,self.lc_labels_dict = read_labels(labels_file)
        self.tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=to_lower) ### Set this to to True for uncased models
        self.model = BertForMaskedLM.from_pretrained(path)
        self.model.eval()
        self.descs = read_descs(DESC_FILE)
        self.top_k = topk
        self.patched = patched
        self.abbrev = abbrev
        self.tokmod  = tokmod
        if (cf.read_config()["LOG_DESCS"] == "1"):
            self.log_descs = True
            self.ci_fp = open("log_ci_predictions.txt","w")
            self.cs_fp = open("log_cs_predictions.txt","w")
        else:
            self.log_descs = False
        self.pos_server_url  = cf.read_config()["POS_SERVER_URL"]
        if (tokmod):
            self.o_vocab_dict,self.l_vocab_dict = read_vocab(vocab_path + "/vocab.txt")
        else:
            self.o_vocab_dict = {}
            self.l_vocab_dict = {}
        #pdb.set_trace()

    def dispatch_request(self,url):
        max_retries = 10
        attempts = 0
        while True:
            try:
                r = requests.get(url,timeout=1000)
                if (r.status_code == 200):
                    return r
            except:
                print("Request:", url, " failed. Retrying...")
            attempts += 1
            if (attempts >= max_retries):
                print("Request:", url, " failed")
                break

    def modify_text_to_match_vocab(self,text):
        ret_arr  = []
        text = text.split()
        for word in text:
            if (word in self.o_vocab_dict):
                ret_arr.append(word)
            else:
                if (word.lower() in self.l_vocab_dict):
                    ret_arr.append(self.l_vocab_dict[word.lower()])
                else:
                    ret_arr.append(word)
        return ' '.join(ret_arr)

    #This is bad hack for prototyping - parsing from text output as opposed to json
    def extract_POS(self,text):
        arr = text.split('\n')
        if (len(arr) > 0):
            start_pos = 0
            for i,line in enumerate(arr):
                if (len(line) > 0):
                    start_pos += 1
                    continue
                else:
                    break
            #print(arr[start_pos:])
            terms_arr = []
            for i,line in enumerate(arr[start_pos:]):
                terms = line.split('\t')
                if (len(terms) == 5):
                    #print(terms)
                    terms_arr.append(terms)
            return terms_arr

    def masked_word_first_letter_capitalize(self,entity):
        arr = entity.split()
        ret_arr = []
        for term in arr:
            if (len(term) > 1 and term[0].islower() and term[1].islower()):
                ret_arr.append(term[0].upper() + term[1:])
            else:
                ret_arr.append(term)
        return ' '.join(ret_arr)


    def gen_single_phrase_sentences(self,terms_arr,span_arr):
        sentence_template = "%s is a entity"
        print(span_arr)
        sentences = []
        singleton_spans_arr  = []
        run_index = 0
        entity  = ""
        singleton_span = []
        while (run_index < len(span_arr)):
            if (span_arr[run_index] == 1):
                while (run_index < len(span_arr)):
                    if (span_arr[run_index] == 1):
                        #print(terms_arr[run_index][WORD_POS],end=' ')
                        if (len(entity) == 0):
                            entity = terms_arr[run_index][utils.WORD_POS]
                        else:
                            entity = entity + " " + terms_arr[run_index][utils.WORD_POS]
                        singleton_span.append(1)
                        run_index += 1
                    else:
                        break
                #print()
                for i in sentence_template.split():
                    if (i != "%s"):
                        singleton_span.append(0)
                entity = self.masked_word_first_letter_capitalize(entity)
                sentence = sentence_template % entity
                sentences.append(sentence)
                singleton_spans_arr.append(singleton_span)
                print(sentence)
                print(singleton_span)
                entity = ""
                singleton_span = []
            else:
                run_index += 1
        return sentences,singleton_spans_arr


    def gen_padded_sentence(self,text,max_tokenized_sentence_length,tokenized_text_arr,orig_tokenized_length_arr,indexed_tokens_arr,attention_mask_arr,to_replace):
        if (to_replace):
            text_arr = text.split()
            new_text_arr = []
            for i in range(len(text_arr)):
                if (text_arr[i] == "entity"):
                    new_text_arr.append( "[MASK]")
                else:
                    new_text_arr.append(text_arr[i])
            text = ' '.join(new_text_arr)
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tok_length = len(indexed_tokens)
        max_tokenized_sentence_length = max_tokenized_sentence_length if tok_length <= max_tokenized_sentence_length else tok_length
        indexed_tokens_arr.append(indexed_tokens)
        attention_mask_arr.append([1]*tok_length)
        tokenized_text_arr.append(tokenized_text)
        orig_tokenized_length_arr.append(tokenized_text)
        return max_tokenized_sentence_length

    #Recreate span arr to account for tokenization splitting words.
    def recreate_span(self,main_sent_arr,tokenized_text,span_arr):
        span_arr_index = 0
        last_span_value = -1
        revised_span_arr = []
        adjusted_tokenized_text = []
        main_arr_len = len(main_sent_arr)
        for i in range(len(tokenized_text)):
            #print(i,tokenized_text[i])
            adjusted_tokenized_text.append(tokenized_text[i])
            if (i == 0 or i == len(tokenized_text) -1):
                revised_span_arr.append(0)
                continue
            if (tokenized_text[i].startswith('##')):
                assert(last_span_value != -1)
                revised_span_arr.append(last_span_value)
            else:
                if (span_arr_index < main_arr_len and main_sent_arr[span_arr_index].startswith(tokenized_text[i])):
                    print(main_sent_arr[span_arr_index],tokenized_text[i],span_arr_index)
                    revised_span_arr.append(span_arr[span_arr_index])
                    last_span_value = span_arr[span_arr_index]
                    span_arr_index += 1
                else:
                    #this is the case where tokenizer breaks down a word without # sign addition. Example BCR-ABL is broken into BCR,-,ABL So we treat as though it is so
                    adjusted_tokenized_text[i] = "##" + tokenized_text[i]
                    revised_span_arr.append(last_span_value)
        assert(len(revised_span_arr) == len(tokenized_text))
        assert(len(adjusted_tokenized_text) == len(tokenized_text))
        return revised_span_arr,adjusted_tokenized_text

    def aggregate_span_word_predictions_in_main_sentence(self,ret_obj,span_arr):
        new_sent_obj = {} 
        last_key_pos = -1
        curr_sent_obj = ret_obj[0]["predictions"]
        for key in curr_sent_obj:
            span_index = key - 1
            if (span_arr[span_index] == 1):
                if (span_index > 0 and span_arr[span_index -1] == 1):
                    new_sent_obj[last_key_pos].extend(curr_sent_obj[key])
                else:
                    last_key_pos = key
                    new_sent_obj[key] = curr_sent_obj[key]
            else:
                if (key in  curr_sent_obj):
                    new_sent_obj[key] = curr_sent_obj[key]
        ret_obj[0]["predictions"] = new_sent_obj
    

    def find_entity(self,word):
        entities = self.labels_dict
        lc_entities = self.lc_labels_dict
        #words = self.filter_glue_words(words) #do not filter glue words anymore. Let them pass through
        l_word = word.lower()
        if l_word.isdigit():
            ret_label = "MEASURE"
            ret_counts = str(1)
        elif (word in entities):
            ret_label = entities[word]["label"]
            ret_counts = entities[word]["counts"]
        elif (l_word in entities):
            ret_label = entities[l_word]["label"]
            ret_counts = entities[l_word]["counts"]
        elif (l_word in lc_entities):
            ret_label = lc_entities[l_word]["label"]
            ret_counts = lc_entities[l_word]["counts"]
        else:
            ret_label = "OTHER"
            ret_counts = "1"
        if (ret_label == "OTHER"):
            ret_label = "UNTAGGED_ENTITY"
            ret_counts = "1"
        print(word,ret_label,ret_counts)
        return ret_label,ret_counts

    #This is just a trivial hack for consistency of CI prediction of numbers
    def override_ci_number_predictions(self,masked_sent):                                                                                                                                                                                                          
        words = masked_sent.split()                                                                                                                                                                                                                        
        words_count = len(words)                                                                                                                                                                                                                           
        if (len(words) == 4 and words[words_count-1] == "entity" and words[words_count -2] == "a" and words[words_count -3] == "is"  and words[0].isnumeric()): #only integers skipped                                                                     
            return True                                                                                                                                                                                                                               
        else:                                                                                                                                                                                                                                              
            return False
                               

    def get_descriptors(self,sent):
        '''
            Batched creation of descriptors given a sentence.
                1) Find noun phrases to tag in a sentence if user did not explicitly tag. 
                2) Create 'N' CI sentences if there are N phrases to tag. So in total 1 + N sentences
                3) Create a batch padding all sentences to the maximum sentence length.
                4) Perform inference on batch 
                5) Return json of descriptors for the ooriginal sentence as well as all CI sentences
        '''
        #This is a modification of input text to words in vocab that match it in case insensitive manner. 
        #This is no longer required when we are using subwords too for prediction.
        if (self.tokmod):
            sent = self.modify_text_to_match_vocab(sent)

        #Step 1. Find entities to tag if user did not explicitly tag terms
        #All noun phrases are tagged for prediction
        if (SPECIFIC_TAG in sent):
            terms_arr = utils.set_POS_based_on_entities(sent)
        else:
            url = self.pos_server_url  + sent.replace('"','\'')
            r = self.dispatch_request(url)
            terms_arr = self.extract_POS(r.text)
    
        #Note span arr only contains phrases in the input that need to be tagged - not the span of all phrases in sentences
        main_sent_arr,masked_sent_arr,span_arr = utils.detect_masked_positions(terms_arr)

        #TBD. check if this is rquired anymore
        not_used_masked_sent_arr,span_arr = utils.filter_common_noun_spans(span_arr,masked_sent_arr,terms_arr,self.descs)

        #Step 2. Create N CI sentences
        singleton_sentences,not_used_singleton_spans_arr = self.gen_single_phrase_sentences(terms_arr,span_arr)

        #We now have 1 + N sentences
        max_tokenized_sentence_length = 0
        tokenized_text_arr = []
        indexed_tokens_arr = []
        attention_mask_arr = []
        all_sentences_arr = []
        orig_tokenized_length_arr = []
        all_sentences_arr.append(' '.join(main_sent_arr))
        max_tokenized_sentence_length = self.gen_padded_sentence(all_sentences_arr[0],max_tokenized_sentence_length,tokenized_text_arr,orig_tokenized_length_arr,indexed_tokens_arr,attention_mask_arr,False)
        for text in singleton_sentences:
            all_sentences_arr.append(text)
            max_tokenized_sentence_length = self.gen_padded_sentence(text,max_tokenized_sentence_length,tokenized_text_arr,orig_tokenized_length_arr,indexed_tokens_arr,attention_mask_arr,True)


        #pad all sentences with length less than max sentence length. This includes the full sentence too since we used indexed_tokens_arr
        for i in range(len(indexed_tokens_arr)):
            padding = [self.tokenizer.pad_token_id]*(max_tokenized_sentence_length - len(indexed_tokens_arr[i]))
            att_padding = [0]*(max_tokenized_sentence_length - len(indexed_tokens_arr[i]))
            if (len(padding) > 0):
                indexed_tokens_arr[i].extend(padding)
                attention_mask_arr[i].extend(att_padding)

        #create revised span arr for main sentence, taking into account tokenization of text
        #We need this to aggregate predictions across tokenized pieces of a word whose descriptors that
        #are of interest
        revised_span_arr,adjusted_tokenized_text = self.recreate_span(main_sent_arr,tokenized_text_arr[0],span_arr) #Recreate the span arr of the main sentence
        

        assert(len(main_sent_arr) == len(span_arr))
        assert(len(all_sentences_arr) == len(indexed_tokens_arr))
        assert(len(all_sentences_arr) == len(attention_mask_arr))
        assert(len(all_sentences_arr) == len(tokenized_text_arr))
        assert(len(all_sentences_arr) == len(orig_tokenized_length_arr))
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens_arr)
        attention_tensors = torch.tensor(attention_mask_arr)


        ret_obj = OrderedDict()
        with torch.no_grad():
            predictions = self.model(tokens_tensor, attention_mask=attention_tensors)
            for sent_index in  range(len(predictions[0])):

                ret_obj[sent_index] = {}
                ret_obj[sent_index]["sentence"] = all_sentences_arr[sent_index]
                print("*** Current sentence ***",all_sentences_arr[sent_index])
                if (self.log_descs):
                    fp = self.cs_fp if sent_index == 0  else self.ci_fp
                    fp.write("\nCurrent sentence: " + all_sentences_arr[sent_index] + "\n")
                curr_sent = {}
                ret_obj[sent_index]["predictions"] = curr_sent
                whole_word_count = 0

                for word in range(len(tokenized_text_arr[sent_index])):
                    if ((word == 0 and sent_index == 0) or word == len(tokenized_text_arr[sent_index]) - 1): # We skip cls for main sentence. We use cls only for CI predictions. SEP is always skipped
                        continue
                    if (sent_index == 0 and  not adjusted_tokenized_text[word].startswith("##")):
                        whole_word_count += 1
                    if (sent_index == 0 and revised_span_arr[word] == 0):
                        continue
#                    if (sent_index != 0 and (word != 0 and word != len(orig_tokenized_length_arr[sent_index]) - 2)): #For all CI sentences pick only the neighbors of CLS and the last word of the sentence (X is a entity)
                    if (sent_index != 0 and (word != 0 and (word == len(orig_tokenized_length_arr[sent_index]) - 4))): #For all CI sentences pick all terms excluding "is" in "X is a entity"
                        continue
                    results_dict = {}
                    masked_index = word
                    #pick all model predictions for current position word
                    if (self.patched):
                        for j in range(len(predictions[0][0][sent_index][masked_index])):
                            tok = tokenizer.convert_ids_to_tokens([j])[0]
                            results_dict[tok] = float(predictions[0][0][sent_index][masked_index][j].tolist())
                    else:
                        for j in range(len(predictions[0][sent_index][masked_index])):
                            tok = self.tokenizer.convert_ids_to_tokens([j])[0]
                            results_dict[tok] = float(predictions[0][sent_index][masked_index][j].tolist())
                    k = 0
                    #sort it - big to small
                    sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))


                    print("********* Top predictions for token: ",tokenized_text_arr[sent_index][word])
                    if (self.log_descs):
                        fp.write("********* Top predictions for token: " + tokenized_text_arr[sent_index][word] + "\n")
                    #pdb.set_trace()
                    #if (sent_index == 0):
                    #    top_k = self.top_k
                    #else:
                    #    top_k = self.top_k/2
                    top_k = self.top_k
                    for index in sorted_d:
                        #if (index in string.punctuation or index.startswith('##') or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                        if index.lower() in self.descs:
                            continue
                        if (index in string.punctuation  or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                            continue
                        #print(index,round(float(sorted_d[index]),4))
                        if (sent_index == 0):
                            if (whole_word_count not in curr_sent):
                                curr_sent[whole_word_count] = []
                            entity,entity_count = self.find_entity(index)
                            if (self.log_descs):
                                self.cs_fp.write(index + " " + entity +  " " +  entity_count + " " + str(round(float(sorted_d[index]),4)) + "\n")
                            curr_sent[whole_word_count].append({"term":tokenized_text_arr[sent_index][word],"desc":index,"e":entity,"e_count":entity_count,"v":str(round(float(sorted_d[index]),4))})
                        else:
                            #CI predictions of the form X is a entity
                            if (whole_word_count not in curr_sent):
                                curr_sent[whole_word_count] = []
                            if (word == 0):
                                term = "CLS"
                            else:
                                term = "entity"
                            entity,entity_count = self.find_entity(index)
                            override = self.override_ci_number_predictions(all_sentences_arr[sent_index])
                            if (override):
                               index = "two"
                               entity_count = "1"
                               entity = "NUMBER"
                            if (self.log_descs):
                                self.ci_fp.write(index + " " + entity + " " +  entity_count + " " + str(round(float(sorted_d[index]),4)) +  "\n")
                            curr_sent[whole_word_count].append({"term":term,"desc":index,"e":entity,"e_count":entity_count,"v":str(round(float(sorted_d[index]),4))})
                        k += 1
                        if (k > top_k):
                            break
                    print()
        print("Input:",sent)
        #print(ret_obj)
        self.aggregate_span_word_predictions_in_main_sentence(ret_obj,span_arr)
        #print(ret_obj)
        #pdb.set_trace()
        #final_obj = {"terms_arr":main_sent_arr,"span_arr":span_arr,"descs_and_entities":ret_obj,"all_sentences":all_sentences_arr}
        final_obj = {"terms_arr":main_sent_arr,"span_arr":span_arr,"descs_and_entities":ret_obj}
        if (self.log_descs):
            self.ci_fp.flush()
            self.cs_fp.flush()
        return final_obj



def test(singleton,test):
    print(test)
    out = singleton.get_descriptors(test)
    print(out)
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT descriptor service given a sentence. The word to be masked is specified as the special token entity ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-topk', action="store", dest="topk", default=DEFAULT_TOP_K,type=int,help='Number of neighbors to display')
    parser.add_argument('-tolower', dest="tolower", action='store_true',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.add_argument('-no-tolower', dest="tolower", action='store_false',help='Convert tokens to lowercase. Set to True only for uncased models')
    parser.set_defaults(tolower=False)
    parser.add_argument('-patched', dest="patched", action='store_true',help='Is pytorch code patched to harvest [CLS]')
    parser.add_argument('-no-patched', dest="patched", action='store_false',help='Is pytorch code patched to harvest [CLS]')
    parser.add_argument('-abbrev', dest="abbrev", action='store_true',help='Just output pivots - not all neighbors')
    parser.add_argument('-no-abbrev', dest="abbrev", action='store_false',help='Just output pivots - not all neighbors')
    parser.add_argument('-tokmod', dest="tokmod", action='store_true',help='Modify input token casings to match vocab - meaningful only for cased models')
    parser.add_argument('-no-tokmod', dest="tokmod", action='store_false',help='Modify input token casings to match vocab - meaningful only for cased models')
    parser.add_argument('-vocab', action="store", dest="vocab", default=DEFAULT_MODEL_PATH,help='Path to vocab file. This is required only if tokmod is true')
    parser.add_argument('-labels', action="store", dest="labels", default=DEFAULT_LABELS_PATH,help='Path to labels file. This returns labels also')
    parser.set_defaults(tolower=False)
    parser.set_defaults(patched=False)
    parser.set_defaults(abbrev=True)
    parser.set_defaults(tokmod=False)

    results = parser.parse_args()
    try:
        singleton = BatchInference(results.model,results.tolower,results.patched,results.topk,results.abbrev,results.tokmod,results.vocab,results.labels)
        print("To lower casing is set to:",results.tolower)
        #out = singleton.punct_sentence("Apocalypse is a entity")
        #print(out)
        test(singleton,"Fyodor:__entity__ Mikhailovich:__entity__ Dostoevsky:__entity__ was treated for Parkinsons")
        test(singleton,"imatinib was used to treat Michael:__entity__ Jackson:__entity__")
        test(singleton,"Ajit flew to Boston:__entity__")
        test(singleton,"Ajit:__entity__ flew to Boston")
        test(singleton,"A eGFR below 60:__entity__ indicates chronic kidney disease")
        test(singleton,"imatinib was used to treat Michael Jackson")
        test(singleton,"Ajit Valath:__entity__ Rajasekharan is an engineer at nFerence headquartered in Cambrigde MA")
        test(singleton,"imatinib:__entity__")
        test(singleton,"imatinib")
        test(singleton,"iplimumab:__entity__")
        test(singleton,"iplimumab")
        test(singleton,"engineer:__entity__")
        test(singleton,"engineer")
        test(singleton,"Complications include peritonsillar:__entity__ abscess::__entity__")
        test(singleton,"Imatinib was the first signal transduction inhibitor (STI), used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic:__entity__ myeloid:__entity__ leukemia:__entity__ (CML)")
        test(singleton,"Imatinib was the first signal transduction inhibitor (STI), used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic myeloid leukemia (CML)")
        test(singleton,"Imatinib was the first signal transduction inhibitor (STI), used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic:__entity__ myeloid:___entity__ leukemia:__entity__ (CML)")
        test(singleton,"Ajit Rajasekharan is an engineer:__entity__ at nFerence:__entity__")
        test(singleton,"Imatinib was the first signal transduction inhibitor (STI), used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic myeloid leukemia (CML)")
        test(singleton,"Ajit:__entity__ Rajasekharan:__entity__ is an engineer")
        test(singleton,"Imatinib:__entity__ was the first signal transduction inhibitor (STI), used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic myeloid leukemia (CML)")
        test(singleton,"Ajit Valath Rajasekharan is an engineer at nFerence headquartered in Cambrigde MA")
        test(singleton,"Ajit:__entity__ Valath Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde MA")
        test(singleton,"Ajit:__entity__ Valath:__entity__ Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde MA")
        test(singleton,"Ajit:__entity__ Valath:__entity__ Rajasekharan:__entity__ is an engineer:__entity__ at nFerence headquartered in Cambrigde MA")
        test(singleton,"Ajit Raj is an engineer:__entity__ at nFerence")
        test(singleton,"Ajit Valath:__entity__ Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde:__entity__ MA")
        test(singleton,"Ajit Valath Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde:__entity__ MA")
        test(singleton,"Ajit Valath Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde MA")
        test(singleton,"Ajit Valath Rajasekharan is an engineer at nFerence headquartered in Cambrigde MA")
        test(singleton,"Ajit:__entity__ Rajasekharan:__entity__ is an engineer at nFerence:__entity__")
        test(singleton,"Imatinib mesylate is used to treat non small cell lung cancer")
        test(singleton,"Imatinib mesylate is used to treat entity")
        test(singleton,"Imatinib is a entity")
        test(singleton,"nsclc is a entity")
        test(singleton,"Ajit Rajasekharan is a entity")
        test(singleton,"ajit rajasekharan is a entity")
        test(singleton,"John Doe is a entity")
        test(singleton,"john doe is a entity")
        test(singleton,"Abubakkar Siddiq is a entity")
        test(singleton,"eGFR is a entity")
        test(singleton,"EGFR is a entity")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
