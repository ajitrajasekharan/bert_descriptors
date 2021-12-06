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
import json

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
        self.always_log_fp = open("CI_LOGS.txt","a")
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
        #print(span_arr)
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
                #print(sentence)
                #rint(singleton_span)
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
                if (text_arr[i] == "entity" ):
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
        #print(word,ret_label,ret_counts)
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
                2) Create 'N' CS and  CI sentences if there are N phrases to tag.  Total 2*N sentences
                3) Create a batch padding all sentences to the maximum sentence length.
                4) Perform inference on batch 
                5) Return json of descriptors for the ooriginal sentence as well as all CI sentences
        '''
        #This is a modification of input text to words in vocab that match it in case insensitive manner. 
        #This is *STILL* required when we are using subwords too for prediction. The prediction quality is still better.
        #An example is Mesothelioma is caused by exposure to asbestos. The quality of prediction is better when Mesothelioma is not split by lowercasing with A100 model
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
        #Step 2. Create N CS sentences
        #This returns masked sentences for all positions
        main_sent_arr,masked_sent_arr,span_arr = utils.detect_masked_positions(terms_arr)


        #Step 2. Create N CI sentences
        singleton_sentences,not_used_singleton_spans_arr = self.gen_single_phrase_sentences(terms_arr,span_arr)


        #We now have 2*N sentences
        max_tokenized_sentence_length = 0
        tokenized_text_arr = []
        indexed_tokens_arr = []
        attention_mask_arr = []
        all_sentences_arr = []
        orig_tokenized_length_arr = []
        assert(len(masked_sent_arr) == len(singleton_sentences))
        for ci_s,cs_s in zip(singleton_sentences,masked_sent_arr):
            all_sentences_arr.append(ci_s)
            max_tokenized_sentence_length = self.gen_padded_sentence(ci_s,max_tokenized_sentence_length,tokenized_text_arr,orig_tokenized_length_arr,indexed_tokens_arr,attention_mask_arr,True)
            cs_s = ' '.join(cs_s).replace("__entity__","entity")
            all_sentences_arr.append(cs_s)
            max_tokenized_sentence_length = self.gen_padded_sentence(cs_s,max_tokenized_sentence_length,tokenized_text_arr,orig_tokenized_length_arr,indexed_tokens_arr,attention_mask_arr,True)


        #pad all sentences with length less than max sentence length. This includes the full sentence too since we used indexed_tokens_arr
        for i in range(len(indexed_tokens_arr)):
            padding = [self.tokenizer.pad_token_id]*(max_tokenized_sentence_length - len(indexed_tokens_arr[i]))
            att_padding = [0]*(max_tokenized_sentence_length - len(indexed_tokens_arr[i]))
            if (len(padding) > 0):
                indexed_tokens_arr[i].extend(padding)
                attention_mask_arr[i].extend(att_padding)


        assert(len(main_sent_arr) == len(span_arr))
        assert(len(all_sentences_arr) == len(indexed_tokens_arr))
        assert(len(all_sentences_arr) == len(attention_mask_arr))
        assert(len(all_sentences_arr) == len(tokenized_text_arr))
        assert(len(all_sentences_arr) == len(orig_tokenized_length_arr))
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens_arr)
        attention_tensors = torch.tensor(attention_mask_arr)

        print("Input:",sent)
        ret_obj = OrderedDict()
        with torch.no_grad():
            predictions = self.model(tokens_tensor, attention_mask=attention_tensors)
            for sent_index in  range(len(predictions[0])):

                #print("*** Current sentence ***",all_sentences_arr[sent_index])
                if (self.log_descs):
                    fp = self.cs_fp if sent_index %2 != 0  else self.ci_fp
                    fp.write("\nCurrent sentence: " + all_sentences_arr[sent_index] + "\n")
                prediction = "ci_prediction" if (sent_index %2 == 0 ) else "cs_prediction"
                out_index = int(sent_index/2) + 1
                if (out_index not in ret_obj):
                    ret_obj[out_index] = {}
                assert(prediction not in ret_obj[out_index])
                ret_obj[out_index][prediction] = {}
                ret_obj[out_index][prediction]["sentence"] = all_sentences_arr[sent_index]
                curr_sent_arr = []
                ret_obj[out_index][prediction]["descs"] = curr_sent_arr

                for word in range(len(tokenized_text_arr[sent_index])):
                    if (word == len(tokenized_text_arr[sent_index]) - 1): # SEP is  skipped for CI and CS
                        continue
                    #if (sent_index != 0 and (word != 0 and word != len(orig_tokenized_length_arr[sent_index]) - 2)): #For all CI sentences pick only the neighbors of CLS and the last word of the sentence (X is a entity)
                    if (sent_index %2 == 0 and (word != 0 and word != len(orig_tokenized_length_arr[sent_index]) - 2) and word != len(orig_tokenized_length_arr[sent_index]) - 3): #For all CI sentences - just pick CLS, "a" and "entity"
                    #if (sent_index != 0 and (word != 0 and (word == len(orig_tokenized_length_arr[sent_index]) - 4))): #For all CI sentences pick ALL terms excluding "is" in "X is a entity"
                        continue
                    if (sent_index %2 != 0 and tokenized_text_arr[sent_index][word] != "[MASK]"): # for all CS sentences skip all terms except the mask position
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


                    #print("********* Top predictions for token: ",tokenized_text_arr[sent_index][word])
                    if (self.log_descs):
                        fp.write("********* Top predictions for token: " + tokenized_text_arr[sent_index][word] + "\n")
                    top_k = self.top_k
                    for index in sorted_d:
                        #if (index in string.punctuation or index.startswith('##') or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                        if index.lower() in self.descs:
                            continue
                        if (index in string.punctuation  or len(index) == 1 or index.startswith('.') or index.startswith('[')):
                            continue
                        #print(index,round(float(sorted_d[index]),4))
                        if (sent_index % 2 != 0):
                            #CS predictions
                            entity,entity_count = self.find_entity(index)
                            if (self.log_descs):
                                self.cs_fp.write(index + " " + entity +  " " +  entity_count + " " + str(round(float(sorted_d[index]),4)) + "\n")
                            curr_sent_arr.append({"desc":index,"e":entity,"e_count":entity_count,"v":str(round(float(sorted_d[index]),4))})
                            if (all_sentences_arr[sent_index].strip().endswith("entity")):
                                self.always_log_fp.write(' '.join(all_sentences_arr[sent_index].split()[:-1]) + " " + index + " :__entity__\n")
                        else:
                            #CI predictions of the form X is a entity
                            entity,entity_count = self.find_entity(index)
                            override = self.override_ci_number_predictions(all_sentences_arr[sent_index])
                            if (override):
                               index = "two"
                               entity_count = "1"
                               entity = "NUMBER"
                            if (self.log_descs):
                                self.ci_fp.write(index + " " + entity + " " +  entity_count + " " + str(round(float(sorted_d[index]),4)) +  "\n")
                            curr_sent_arr.append({"desc":index,"e":entity,"e_count":entity_count,"v":str(round(float(sorted_d[index]),4))})
                            #if (index != "two" and not index.startswith("#")  and not all_sentences_arr[sent_index].strip().startswith("is ")):
                            if (index != "two" and not all_sentences_arr[sent_index].strip().startswith("is ")):
                                self.always_log_fp.write(' '.join(all_sentences_arr[sent_index].split()[:-1]) + " " + index + " :__entity__\n")
                        k += 1
                        if (k >= top_k):
                            break
                    #print()
        #print(ret_obj)
        #print(ret_obj)
        #pdb.set_trace()
        #final_obj = {"terms_arr":main_sent_arr,"span_arr":span_arr,"descs_and_entities":ret_obj,"all_sentences":all_sentences_arr}
        final_obj = {"input":sent,"terms_arr":main_sent_arr,"span_arr":span_arr,"descs_and_entities":ret_obj}
        if (self.log_descs):
            self.ci_fp.flush()
            self.cs_fp.flush()
        self.always_log_fp.flush()
        return final_obj


test_arr = [
        "Mesothelioma is caused by exposure to asbestos:__entity__",
        "Asbestos is a highly :__entity__",
        "Fyodor:__entity__ Mikhailovich:__entity__ Dostoevsky:__entity__ was treated for Parkinsons:__entity__ and later died of lung carcinoma",
        "Fyodor:__entity__ Mikhailovich:__entity__ Dostoevsky:__entity__",
        "imatinib was used to treat Michael:__entity__ Jackson:__entity__",
        "Ajit flew to Boston:__entity__",
        "Ajit:__entity__ flew to Boston",
        "A eGFR below 60:__entity__ indicates chronic kidney disease",
        "imatinib was used to treat Michael Jackson",
        "Ajit Valath:__entity__ Rajasekharan is an engineer at nFerence headquartered in Cambrigde MA",
        "imatinib:__entity__",
        "imatinib",
        "iplimumab:__entity__",
        "iplimumab",
        "engineer:__entity__",
        "engineer",
        "Complications include peritonsillar:__entity__ abscess::__entity__",
        "Imatinib was the first signal transduction inhibitor (STI,, used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic:__entity__ myeloid:__entity__ leukemia:__entity__ (CML,",
        "Imatinib was the first signal transduction inhibitor (STI,, used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic myeloid leukemia (CML,",
        "Imatinib was the first signal transduction inhibitor (STI,, used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic:__entity__ myeloid:___entity__ leukemia:__entity__ (CML,",
        "Ajit Rajasekharan is an engineer:__entity__ at nFerence:__entity__",
        "Imatinib was the first signal transduction inhibitor (STI,, used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic myeloid leukemia (CML,",
        "Ajit:__entity__ Rajasekharan:__entity__ is an engineer",
        "Imatinib:__entity__ was the first signal transduction inhibitor (STI,, used in a clinical setting. It prevents a BCR-ABL protein from exerting its role in the oncogenic pathway in chronic myeloid leukemia (CML,",
        "Ajit Valath Rajasekharan is an engineer at nFerence headquartered in Cambrigde MA",
        "Ajit:__entity__ Valath Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde MA",
        "Ajit:__entity__ Valath:__entity__ Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde MA",
        "Ajit:__entity__ Valath:__entity__ Rajasekharan:__entity__ is an engineer:__entity__ at nFerence headquartered in Cambrigde MA",
        "Ajit Raj is an engineer:__entity__ at nFerence",
        "Ajit Valath:__entity__ Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde:__entity__ MA",
        "Ajit Valath Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde:__entity__ MA",
        "Ajit Valath Rajasekharan is an engineer:__entity__ at nFerence headquartered in Cambrigde MA",
        "Ajit Valath Rajasekharan is an engineer at nFerence headquartered in Cambrigde MA",
        "Ajit:__entity__ Rajasekharan:__entity__ is an engineer at nFerence:__entity__",
        "Imatinib mesylate is used to treat non small cell lung cancer",
        "Imatinib mesylate is used to treat :__entity__",
        "Imatinib is a term:__entity__",
        "nsclc is a term:__entity__",
        "Ajit Rajasekharan is a term:__entity__",
        "ajit rajasekharan is a term:__entity__",
        "John Doe is a term:__entity__"
]

def test_sentences(singleton,iter_val):
    with open("debug.txt","w") as fp:
        for test in iter_val:
            test = test.rstrip('\n')
            fp.write(test + "\n")
            print(test)
            out = singleton.get_descriptors(test)
            print(out)
            fp.write(json.dumps(out,indent=4))
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT descriptor service given a sentence. The word to be masked is specified as the special token entity ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input", default="",help='Optional input file with sentences. If not specified, assumed to be canned sentence run (default behavior)')
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
    parser.set_defaults(tokmod=True)

    results = parser.parse_args()
    try:
        singleton = BatchInference(results.model,results.tolower,results.patched,results.topk,results.abbrev,results.tokmod,results.vocab,results.labels)
        print("To lower casing is set to:",results.tolower)
        if (len(results.input) == 0):
            print("Canned test mode")
            test_sentences(singleton,test_arr)
        else:
            print("Batch file test mode")
            fp = open(results.input)
            test_sentences(singleton,fp)
                
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)

