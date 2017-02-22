## KDD_linelisting
This repository contains the source codes and data for the paper
entitled ``Guided Deep List: Automating the Generation of Epidemiological Line
Lists from Open Sources`` communicated to ACM SIGKDD 2017. The code for the proposed 
model 'Guided Deep List' is located at [linelist_code](./code/ll_code.py) and the code 
for the baseline model 'Guidedlist' is located at [baseline_code](./code/ll_baseline.py). 
The codes can be executed as follows.

Code: [linelist_code](./code/ll_code.py)
usage: Automated line listing [-h] -i MERSBULLETINS -v WHOVEC -ind NUMIND -o
                              OUTPUTLL

optional arguments:
  -h, --help            show this help message and exit
  -i MERSBULLETINS, --MERSbulletins MERSBULLETINS
                        Input file containing the WHO MERS bulletins from
                        which line list will be extracted
  -v WHOVEC, --whovec WHOVEC
                        word vectors corresponding to the WHO corpus
  -ind NUMIND, --numind NUMIND
                        Number of indicators to be used for extracting line
                        list features
  -o OUTPUTLL, --outputll OUTPUTLL
                        File where the automatically extracted line list will
                        be dumped

Example: python ``./code/ll_code.py`` -i ``./data/WHO_KSA_MERS_bulletins.json`` -v
``./data/WHO_vectors/WHO_SGHS_vectors.word2vec`` -ind ``7`` -o
``./data/automated_ll/automated_ll_KSA_SGHS.json``

Code: [baseline_code](./code/ll_baseline.py)
usage: Baseline line listing [-h] -i MERSBULLETINS -o OUTPUTLL

optional arguments:
  -h, --help            show this help message and exit
  -i MERSBULLETINS, --MERSbulletins MERSBULLETINS
                        Input file containing the WHO MERS bulletins from
                        which line list will be extracted
  -o OUTPUTLL, --outputll OUTPUTLL
                        File where the automatically extracted line list will
                        be dumped

Example: python ``./code/ll_baseline.py`` -i
``./data/WHO_KSA_MERS_bulletins.json`` -o
``./data/automated_ll/automated_ll_baseline.json``


## Confusion matrices 

The confusion matrices corresponding to the performance of each model for a
clinical feature can be found at [confusion_matrix](./data/confusion_matrix/).


