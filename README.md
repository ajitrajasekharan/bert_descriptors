# bert_descriptors
BERT's MLM head model exposed as a HTTP service

*Example below uses BERT large cased*

This service is used by [Unsupervised NER](https://github.com/ajitrajasekharan/unsupervised_NER)
# Installation

Setup pytorch environment with/without GPU support using link [Multi GPU test](https://github.com/ajitrajasekharan/multi_gpu_test)

*Make sure to follow conda environment activation instructions. Also tensorflow is not required. So we just neeed to run first.sh, second.sh and third.sh (from 1-4) performing the instructions displayed at end of each step *


# Usage 
Download a BERT model from Hugginface.

*e.g. [BERT large cased - pytorch_model.bin,vocab.txt,config.json](https://huggingface.co/bert-large-uncased/tree/main) into current directory*

Start server

  $ ./run_server.sh
  

Confirm installation works by

$ wget -O DES "http://127.0.0.1:8087/dummy/0/John flew from entity to Rio De Janiro"

The output DES file should contain


<img src="DES.png" width="600">
  
  
 # Revision notes
 
 17 Sept 2021
 
 -  Two kinds of descriptor sets are returned now. 
      1) Return just the descriptors of the masked position as before (option 0 in GET call)
      2) Return descriptors of masked position as well as for [CLS] position (option 1 in GET call). This is useful for detecting entity type of a term/phrase. For example: Imatinib is a entity
      3) Common descriptors are now used to filter terms that do not convey any entity information in results. This is to maximize entity information in generated output
  
 - Adaptive casing of input tokens to maximize hit with underlying vocab. This is meaningful for cased models only
 - Full distribution output is turned off by default and needs to be enabled explicitly. The output above is with full distribution option turned on.
 
 
 # License
 
 MIT License

