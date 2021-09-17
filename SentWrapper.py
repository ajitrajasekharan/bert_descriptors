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

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


DEFAULT_TOP_K = 40
DEFAULT_MODEL_PATH='./'
DEFAULT_TO_LOWER=False
DESC_FILE="./common_descs.txt"

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

                
class SentWrapper:
    def __init__(self, path,to_lower,patched,topk,abbrev,tokmod,vocab_path,usecls):
        print("Model path:",path,"lower casing set to:",to_lower," is patched ", patched)
        self.path = path
        self.tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=to_lower) ### Set this to to True for uncased models
        self.model = BertForMaskedLM.from_pretrained(path)
        self.model.eval()
        self.descs = read_descs(DESC_FILE)
        self.top_k = topk
        self.patched = patched
        self.abbrev = abbrev
        self.tokmod  = tokmod
        self.usecls  = usecls
        if (tokmod):
            self.o_vocab_dict,self.l_vocab_dict = read_vocab(vocab_path + "/vocab.txt")
        else:
            self.o_vocab_dict = {}
            self.l_vocab_dict = {}
        #pdb.set_trace()

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

    def punct_sentence(self,text,usecls):

        if (self.tokmod):
            text = self.modify_text_to_match_vocab(text)
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        masked_index = 0
        original_masked_index = 0

        for i in range(len(tokenized_text)):
            if (tokenized_text[i] == "entity"):
                masked_index = i
                original_masked_index = i
                break
        #assert (masked_index != 0)
        if (masked_index == 0):
            return "Specify and input sentence with the term entity in it. This word will be masked"
        tokenized_text[masked_index] = "[MASK]"
        indexed_tokens[masked_index] = self.tokenizer.convert_tokens_to_ids("[MASK]")
        print(tokenized_text)
        print(masked_index)
        results_dict = {}

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        ret_str = ""
        mask_str = ""
        debug_str = "\nPIVOT_DESCRIPTORS:"
        cls_str = ""
        addl_debug_str = ""
        delimiter_str = "\n--------Neighbors for all words in sentence below (including MASK word)-----\n\n"
        head_str ="\nTokenized input:" +  ' '.join(tokenized_text) + "\n\n"
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            for word in range(len(tokenized_text)):
                if (word == len(tokenized_text) -1):
                    continue
                masked_index = word
                ret_str += "\n\n" + str(word) + ") Neighbors for word: "  + tokenized_text[word] + "\n"
                if (original_masked_index == word):
                    mask_str += "\n\n" + str(word) + ") Neighbors for word: "  + tokenized_text[word] + "\n"
                if (self.patched):
                    arr = np.array(predictions[0][0][0,masked_index].tolist())
                else:
                    arr = np.array(predictions[0][0][masked_index].tolist())
                mean = np.mean(arr)
                std = np.std(arr)
                min_val = np.min(arr)
                max_val = np.max(arr)
                hist,bins = np.histogram(arr,bins = [-50,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,50])
                #hist,bins = np.histogram(arr,bins = [-20,-16,-12,-8,-4,0,4,8,12,16,20])
                if (original_masked_index == word):
                    mask_str += "Stats:" + " mean: " + str(mean) + " std: " + str(std) + " max: " + str(max_val) + " min: " + str(min_val)  +"\n"
                ret_str += "Stats:" + " mean: " + str(mean) + " std: " + str(std) + " max: " + str(max_val) + " min: " + str(min_val)  +"\n"
                bin_str = ""
                assert len(bins) == len(hist) + 1
                for bin_index in range(len(bins)):
                    if (bin_index < len(hist)):
                        bin_str +=  " [" + str(bins[bin_index]) + " to " + str(bins[bin_index+1]) + ") :" + str(hist[bin_index]) + "\n"
                    #else:
                    #    bin_str +=  " " + str(bins[bin_index])
                #hist_str = ""
                #for hist_index in range(len(hist)):
                #    hist_str +=  " " + str(hist[hist_index])
                if (original_masked_index == word):
                    mask_str += "Bins-counts:\n" +  bin_str + "\n"
                ret_str += "Bins-counts:\n" +  bin_str + "\n"
                #ret_str += "Hist: " + str(hist_str) + "\n"
                if (self.patched):
                    for i in range(len(predictions[0][0][0,masked_index])):
                        tok = self.tokenizer.convert_ids_to_tokens([i])[0]
                        results_dict[tok] = float(predictions[0][0][0,masked_index][i].tolist())
                else:
                    for i in range(len(predictions[0][0][masked_index])):
                        tok = self.tokenizer.convert_ids_to_tokens([i])[0]
                        results_dict[tok] = float(predictions[0][0][masked_index][i].tolist())
                k = 0
                sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
                debug_count = 0
                cls_count = 0
                for j in sorted_d:
                    if (j.lower() in self.descs): #eliminate words that gove no information on entity
                        continue
                    if (j in string.punctuation or j.startswith('##') or len(j) == 1 or j.startswith('.') or j.startswith('[')):
                         continue
                    if (original_masked_index == word):
                         mask_str = mask_str + str(k+1) + "] " +  j + " " + str(sorted_d[j]) + "\n"
                         if (debug_count < 10):
                             debug_str  = debug_str + " " + j
                             debug_count += 1
                         else:
                             if (debug_count < 20):
                                addl_debug_str  = addl_debug_str + " " + j
                                debug_count += 1
                    else:
                        if (usecls and word == 0):
                            if (cls_count < 10):
                                cls_str  = cls_str + " " + j
                                cls_count += 1
                    ret_str = ret_str + str(k+1) + "] " +  j + " " + str(sorted_d[j]) + "\n"
                    k += 1
                    if (k >= self.top_k):
                        break
        if (self.abbrev):
            final_str = debug_str
            if (usecls):
                final_str = final_str + " " + cls_str
            else:
                final_str = final_str + " " + addl_debug_str
        else:
            final_str =  head_str + "\n"  + mask_str + debug_str + "\n" +   delimiter_str + ret_str
        return final_str


def test(singleton,test,usecls):
    print(test)
    out = singleton.punct_sentence(test,usecls)
    print(out)


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
    parser.add_argument('-usecls', dest="usecls", action='store_true',help='Use neighbors of [CLS] vector')
    parser.add_argument('-no-usecls', dest="usecls", action='store_false',help='Use neighbors of [CLS] vector')
    parser.set_defaults(tolower=False)
    parser.set_defaults(patched=False)
    parser.set_defaults(abbrev=True)
    parser.set_defaults(tokmod=True)
    parser.set_defaults(usecls=True)

    results = parser.parse_args()
    try:
        singleton = SentWrapper(results.model,results.tolower,results.patched,results.topk,results.abbrev,results.tokmod,results.vocab,results.usecls) 
        print("To lower casing is set to:",results.tolower)
        #out = singleton.punct_sentence("Apocalypse is a entity")
        #print(out)
        test(singleton,"Imatinib mesylate is used to treat entity",False)
        test(singleton,"Imatinib is a entity",singleton.usecls)
        test(singleton,"nsclc is a entity",singleton.usecls)
        test(singleton,"Ajit Rajasekharan is a entity",singleton.usecls)
        test(singleton,"ajit rajasekharan is a entity",singleton.usecls)
        test(singleton,"John Doe is a entity",singleton.usecls)
        test(singleton,"john doe is a entity",singleton.usecls)
        test(singleton,"Abubakkar Siddiq is a entity",singleton.usecls)
        test(singleton,"eGFR is a entity",singleton.usecls)
        test(singleton,"EGFR is a entity",singleton.usecls)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
