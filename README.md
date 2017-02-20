# KDD_linelisting
This repository contains the source codes and data for KDD line listing paper.
The code for the proposed model 'Guided Deep List' is located at [linelist_code](./code/ll_code.py) 
and the code for the baseline model 'Guidedlist' is located at
[baseline_code](./code/ll_baseline.py). The codes can be executed as follows.

* python ./code/ll_code.py ./data/WHO_KSA_MERS_bulletins.json
  ./data/WHO_vectors/WHO_SGNS_vectors.word2vec 5
  ./data/automated_ll/automated_ll_KSA_SGNS.json 

* python ./code/ll_code.py ./data/WHO_KSA_MERS_bulletins.json
  ./data/WHO_vectors/WHO_SGHS_vectors.word2vec 7
  ./data/automated_ll/automated_ll_KSA_SGHS.json 

* python ./code/ll_baseline.py ./data/WHO_KSA_MERS_bulletins.json
  ./data/automated_ll/automated_ll_baseline.json

The confusion matrices corresponding to the performance of each model for a
clinical feature can be found at [confusion_matrix](./data/confusion_matrix/).


