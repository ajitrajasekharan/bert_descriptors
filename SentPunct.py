# -*- coding: utf-8 -*-
import os
import ResponseHandler
import subprocess
import urllib
import SentWrapper
import pdb

#MODEL_PATH ='bert-large-cased'
MODEL_PATH ='./'
TOLOWER = False
PATCHED = False
TOPK = 40
ABBREV = True
TOKMOD = True
VOCAB="./"
USECLS=True #this is however can be overridden on each call 

singleton = None
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class SentPunct(ResponseHandler.ResponseHandler):
    def __init__(self):
        print("This is the constructor method.")
    def handler(self,write_obj = None):
        print("In derived class")
        global singleton
        if singleton is None:
            singleton = SentWrapper.SentWrapper(MODEL_PATH,TOLOWER,PATCHED,TOPK,ABBREV,TOKMOD,VOCAB,USECLS) 
        if (write_obj is not None):
            param =write_obj.path[1:]
            print("Orig Arg = ",param)
            param = '/'.join(param.split('/')[1:])
            print("API param removed Arg = ",param)
            param = urllib.parse.unquote(param).split('/')
            usecls = True if param[0] == '1' else False
            out = singleton.punct_sentence(param[1],usecls)
            #print("Arg = ",write_obj.path[1:])
            #out = singleton.punct_sentence(urllib.parse.unquote(write_obj.path[1:].lower()))
            print(out)
            if (len(out) >= 1):
                write_obj.wfile.write(out.encode())
            else:
                write_obj.wfile.write("0".encode())
            write_obj.wfile.write("\nNF_EOS\n".encode())








def my_test():
    cl = EntityFilter()

    cl.handler()




#my_test()
