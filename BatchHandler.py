# -*- coding: utf-8 -*-
import os
import ResponseHandler
import subprocess
import urllib
import BatchInference
import pdb
import json

#MODEL_PATH ='bert-large-cased'
MODEL_PATH ='./'
DEFAULT_LABELS_PATH='./labels.txt'
TOLOWER = False
PATCHED = False
TOPK = 20
ABBREV = True
#Tok mod is helpful despite using subwords
TOKMOD = True
VOCAB="./"

singleton = None
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class BatchHandler(ResponseHandler.ResponseHandler):
    def __init__(self):
        print("This is the constructor method.")
    def handler(self,write_obj = None):
        print("In derived class")
        global singleton
        if singleton is None:
            singleton = BatchInference.BatchInference(MODEL_PATH,TOLOWER,PATCHED,TOPK,ABBREV,TOKMOD,VOCAB,DEFAULT_LABELS_PATH) 
        if (write_obj is not None):
            param =write_obj.path[1:]
            print("Orig Arg = ",param)
            param = '/'.join(param.split('/')[1:])
            print("API param removed Arg = ",param)
            param = urllib.parse.unquote(param).split('/')
            usecls = True if param[0] == '1' else False #this is a NOOP for batch service
            out = singleton.get_descriptors('/'.join(param[1:]))
            out = json.dumps(out,indent=4)
            #print("Arg = ",write_obj.path[1:])
            #out = singleton.punct_sentence(urllib.parse.unquote(write_obj.path[1:].lower()))
            print(out + "\n\n")
            if (len(out) >= 1):
                write_obj.wfile.write(out.encode())
            else:
                write_obj.wfile.write("0".encode())
            write_obj.wfile.flush()
            #write_obj.wfile.write("\nNF_EOS\n".encode())








def my_test():
    cl = EntityFilter()

    cl.handler()




#my_test()
