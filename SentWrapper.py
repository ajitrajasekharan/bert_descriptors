import torch
import subprocess
from pytorch_transformers import *
import pdb
import operator
from collections import OrderedDict
import numpy as np

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


top_k = 40
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

class SentWrapper:
    def __init__(self, path):
        self.path = path
        self.tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=False) ### Set this to to True for uncased models
        self.model = BertForMaskedLM.from_pretrained(path)
        self.model.eval()
        self.descs = read_descs(DESC_FILE)
        #pdb.set_trace()


    def punct_sentence(self,text):

        text = '[CLS]' + text + '[SEP]'
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
        indexed_tokens[masked_index] = 103
        print(tokenized_text)
        print(masked_index)
        results_dict = {}

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        ret_str = ""
        mask_str = ""
        debug_str = "\nPIVOT_DESCRIPTORS:"
        delimiter_str = "\n--------Neighbors for all words in sentence below (including MASK word)-----\n\n"
        head_str ="\nTokenized input:" +  ' '.join(tokenized_text) + "\n\n"
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            for word in range(len(tokenized_text)):
                if (word == 0 or word == len(tokenized_text) -1):
                    continue
                masked_index = word
                ret_str += "\n\n" + str(word) + ") Neighbors for word: "  + tokenized_text[word] + "\n"
                if (original_masked_index == word):
                    mask_str += "\n\n" + str(word) + ") Neighbors for word: "  + tokenized_text[word] + "\n"
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
                for i in range(len(predictions[0][0][masked_index])):
                    tok = self.tokenizer.convert_ids_to_tokens([i])[0]
                    results_dict[tok] = float(predictions[0][0][masked_index][i].tolist())
                k = 0
                sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
                debug_count = 0
                for j in sorted_d:
                    if (j not in self.descs):
                        continue
                    if (original_masked_index == word):
                         mask_str = mask_str + str(k+1) + "] " +  j + " " + str(sorted_d[j]) + "\n"
                         if (debug_count < 10):
                             debug_str  = debug_str + " " + j
                             debug_count += 1
                    ret_str = ret_str + str(k+1) + "] " +  j + " " + str(sorted_d[j]) + "\n"
                    k += 1
                    if (k >= top_k):
                        break
        return head_str + "\n"  + mask_str + debug_str + "\n" +   delimiter_str + ret_str



def main():
    MODEL_PATH='bert-large-cased'
    singleton = SentWrapper(MODEL_PATH)
    out = singleton.punct_sentence("Apocalypse is a entity")
    print(out)


if __name__ == '__main__':
    main()

