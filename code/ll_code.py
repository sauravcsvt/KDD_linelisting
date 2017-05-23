#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import json
import io
from spacy.tokens.doc import Doc
import spacy.en
import re
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import sys
import networkx as nx
import numpy as np
import nltk
import argparse


class LineList(object):
    """
    Creating a linelist class for extracting line list features corresponding to each infected case

    """
    def __init__(self):
        # Regular expressions to be used for extracting age and gender
        # corresponding to each infected case
        self.regex1 = re.compile(r'.*\s+(?P<age>\d{1,2})(.{0,20})(\s+|-)(?P<gender>woman|man|male|female|boy|girl|housewife).*')
        self.regex2 = re.compile(r'.*\s+(?P<age>\d{1,2})\s*years?(\s|-)old.*')
        self.regex3 = re.compile(r'.*\s*(?P<gender>woman|man|male|female|boy|girl|housewife|he|she).*')
        # Negation cues to be used for extracting clinical features
        # corresponding to each infected case
        self.negation_cues = ["no", "not", "none", "cannot", "without", "lack", "absence"]
        # Creating a regular expression object for checking whether a
        # predictor keyword contains a digit or not 
        self._digits = re.compile('\d')

    # Function to check whether a predictor keyword contains a digit or not 
    def contains_digits(self, word):
        return bool(self._digits.search(word))

    # Function to extract age and gender for each infected case using a
    # hierarchy of regular expressions
    def get_age_gender(self, sent):

        m1 = self.regex1.match(sent)
        if m1:
            age = int(m1.groupdict()['age'])
            gender = m1.groupdict()['gender']
        else:
            m2 = self.regex2.match(sent)
            if m2:
                age = int(m2.groupdict()['age'])
            else:
                age = None

            m3 = self.regex3.match(sent)
            if m3:
                gender = m3.groupdict()['gender']
            else:
                gender = None
        return {'age': age, 'gender': self.get_gender_value(gender)}

    # Function to convert gender expressions, such as 'man', 'woman', 'male',
    # 'female' to binary outcome, such as 'M' or 'F'
    def get_gender_value(self, gender):

        if gender:
            gender = gender.strip(r'\s*|-')
            if gender in ['man', 'male', 'boy', 'he']:
                return 'M'
            elif gender in ['woman', 'female', 'girl', 'housewife', 'she']:
                return 'F'
        return None

    # Function to extract date features, such as onset date, hospitalized date
    # for each infected case
    def infer_date(self, **kwargs):

        predictors = []
        predictor_dist = {}
        predictors.append(kwargs['seed'])
        # Extracting the top-K predictor keywords closer to the seed keyword in vector
        # space using word vectors extracted by word2vec models (SGNS or SGHS). We reject those
        # uninformative keywords which contain a digit
        for similar_word in kwargs['w2v'].most_similar(kwargs['seed'], topn=20):
            if len(similar_word[0]) >= 3 and not self.contains_digits(similar_word[0]):
                predictors.append(similar_word[0])
            if len(predictors) == kwargs['K'] + 1:
                break
        for predictor in predictors:
            predictor_dist[predictor] = {}
            for dt_str in kwargs['dt_dict']:
                predictor_dist[predictor][dt_str] = []
        sent_start_iter = kwargs['start']
        # Iterating over each sentence and calculating the undirected dependency
        # distance from a predictor keyword to a date phrase
        while sent_start_iter <= kwargs['end']:
            sent_dg = nx.Graph()
            for token in kwargs['ll_sents'][sent_start_iter]:
                if token.head.orth_ != token.orth_:
                    sent_dg.add_edge(token.head, token)
            for predictor in predictor_dist:
                for dt_str in kwargs['dt_dict']:
                    is_pred = 0
                    is_dt = 0
                    try:
                        for nd in sent_dg.nodes():
                            if nd.orth_ == predictor:
                                pred_obj = nd
                                is_pred = 1
                            elif nd.orth_ == dt_str:
                                dt_obj = nd
                                is_dt = 1
                        if is_pred and is_dt:
                            # Calculating undirected dependency distance along the
                            # shortest path from the predictor keyword to the date phrase 
                            predictor_dist[predictor][dt_str].append(nx.shortest_path_length(sent_dg, pred_obj, dt_obj))
                    except Exception:
                        continue
            sent_start_iter += 1
        predictor_forecast = {}
        # Extracting the date phrase for each predictor which lies at the shortest
        # dependency distance
        for predictor in predictor_dist:
            predictor_forecast[predictor] = ""
            for dt_str in kwargs['dt_dict']:
                if len(predictor_dist[predictor][dt_str]) == 0:
                    predictor_dist[predictor].pop(dt_str)
                else:
                    predictor_dist[predictor][dt_str] = np.min(predictor_dist[predictor][dt_str])
            if len(predictor_dist[predictor]) > 0:
                predictor_forecast[predictor] = min(predictor_dist[predictor].items(), key=lambda x: x[1])[0]
            if predictor_forecast[predictor] == "":
                predictor_forecast.pop(predictor)
        # extracting the final date phrase output obtained by majority voting
        # on the outputs of the predictors
        try:
            final_forecast = kwargs['dt_dict'][nltk.FreqDist(predictor_forecast.values()).max()]
        except Exception:
            final_forecast = None
        predictor_final = {}
        for predictor in predictors:
            if predictor not in predictor_forecast:
                predictor_final[predictor] = None
            else:
                predictor_final[predictor] = kwargs['dt_dict'][predictor_forecast[predictor]]
        predictor_final["overall"] = final_forecast
        return {"final": final_forecast, "predictor": predictor_final}

    # Function to extract clinical features for each infected case
    def infer_clinical(self, **kwargs):

        predictors = []
        predictor_forecast = defaultdict()
        predictors.append(kwargs['seed'])
        # Extracting the top-K predictor keywords closer to the seed keyword in vector
        # space using word embeddings extracted by word2vec models (SGNS or SGHS). We reject those
        # uninformative keywords which contain a digit
        for similar_word in kwargs['w2v'].most_similar(kwargs['seed'], topn=20):
            if len(similar_word[0]) >= 3 and not self.contains_digits(similar_word[0]):
                predictors.append(similar_word[0])
            if len(predictors) == kwargs['K'] + 1:
                break
        # Extracting the output ('Y' or 'N') of each predictor using direct or
        # indirect negation detection
        for predictor in predictors:
            sent_start_iter = kwargs['start']
            while sent_start_iter <= kwargs['end']:
                is_detect = 0
                sent_dg = nx.DiGraph()
                for token in kwargs['ll_sents'][sent_start_iter]:
                    if token.head.orth_ != token.orth_:
                        sent_dg.add_edges_from([(token.head, token)])
                for nd in sent_dg.nodes():
                    if nd.orth_ == predictor:
                        is_detect = 1
                        is_negation = 0
                        # Perform direct negation detection
                        nd_neighbors = list(set(sent_dg.neighbors(nd)))
                        for neigh_elm in nd_neighbors:
                            if neigh_elm.orth_ in self.negation_cues:
                                is_negation = 1
                        # If no direct negation, perform indirect negation detection
                        if not is_negation:
                            for nd_elm in sent_dg.nodes():
                                try:
                                    if nx.shortest_path(sent_dg, nd_elm, nd):
                                        nd_elm_neighbors = list(set(sent_dg.neighbors(nd_elm)))
                                        for neigh_elm in nd_elm_neighbors:
                                            if neigh_elm.orth_ in self.negation_cues:
                                                is_negation = 1
                                except Exception:
                                    continue
                        if is_negation:
                            predictor_forecast[predictor] = 'N'
                            break
                        else:
                            predictor_forecast[predictor] = 'Y'
                            break
                if not is_detect:
                    sent_start_iter += 1
                else:
                    break
        # Final output ('Y' or 'N') obtained by majority voting on the
        # outputs of the predictors 
        if len(predictor_forecast) == 0:
            final_forecast = 'N'
        else:
            final_forecast = nltk.FreqDist(predictor_forecast.values()).max()
        predictor_final = defaultdict()
        for predictor in predictors:
            if predictor not in predictor_forecast:
                predictor_final[predictor] = 'N'
            else:
                predictor_final[predictor] = predictor_forecast[predictor]
        predictor_final["overall"] = final_forecast
        return {"final": final_forecast, "predictor": predictor_final}


def parse_args():
    '''
    Reads the command line options and parses the appropriate commands
    '''

    ap = argparse.ArgumentParser("Automated line listing")

    # Required Program Arguments
    ap.add_argument("-i", "--MERSbulletins", type=str, required=True,
                    help="Input file containing the WHO MERS bulletins from which line list will be extracted")
    ap.add_argument("-v", "--whovec", type=str, required=True, 
                    help="word vectors corresponding to the WHO corpus")
    ap.add_argument("-ind", "--numind", type=str, required=True, help="Number of predictors to be used for extracting each line list feature")
    ap.add_argument("-o", "--outputll", type=str, required=True, help="File where the automatically extracted line list will be dumped")
    return ap.parse_args()


def main():

    _arg = parse_args()
    # list of articles from which line list features are to be extracted
    # for each infected case
    ll_articles = [json.loads(l) for l in io.open(_arg.MERSbulletins, "r")]
    # word embeddings specific to WHO corpus extracted by word2vec models (SGNS
    # or SGHS)
    w2v_model = Word2Vec.load(_arg.whovec)
    # Number of predictor keywords (excluding the seed keyword)
    K = np.int(_arg.numind)
    # Seed keywords for each line list feature guiding the extraction process 
    seed_keywords = {"Onset Date": "onset", 
                     "Hospital Date": "hospitalized", 
                     "Outcome Date": "died", 
                     "Specified Proximity to Animals or Animal Products": "animals", 
                     "Specified Contact with Other Cases": "case", 
                     "Specified HCW": "healthcare",
                     "Specified Comorbidities": "comorbidities"}
    auto_ll = []
    ll_extract = LineList()
    # Extracting the number of infected cases and the line list features
    # corresponding to each case from each article
    for ll_artl in ll_articles:
        ll_text = ""
        dt_offsets = []
        dt_dict = defaultdict()
        # dt_dict: mapping date phrases to proper datetime strings, e.g.  
        for dtphrase_elm in ll_artl['eventSemantics']['datetimes']:
            dt_dict["-".join(dtphrase_elm['phrase'].split())] = dtphrase_elm["date"]
            dt_offsets.append({'start': ll_artl['BasisEnrichment']['tokens'][int(dtphrase_elm['offset'].split(":")[0])]['start'], 
                               'end': ll_artl['BasisEnrichment']['tokens'][int(dtphrase_elm['offset'].split(":")[1]) - 1]['end']})
        for i in xrange(len(ll_artl["content"])):
            is_offset = 0
            for offset_elm in dt_offsets:
                if int(offset_elm['start']) <= i < int(offset_elm['end']):
                    is_offset = 1
                    if ll_artl["content"][i] == " ":
                        ll_text += "-"
                    else:
                        ll_text += ll_artl["content"][i] 
            if not is_offset:
                if ll_artl["content"][i] == ".":
                    ll_text += " "
                ll_text += ll_artl["content"][i]
        num_cases = [] 
        en_nlp = spacy.load('en')
        ll_doc = Doc(en_nlp.vocab)
        ll_doc = en_nlp(ll_text)
        ll_sents = []
        for sent in ll_doc.sents:
            ll_sents.append(sent)
        # Extracting the number of cases mentioned in the article
        # using age and gender information
        for sent_ind in xrange(len(ll_sents)):
            ag_out = ll_extract.get_age_gender(ll_sents[sent_ind].text)
            if ag_out['age'] is not None and ag_out['gender'] is not None:
                case_feature = defaultdict()
                case_feature['age'] = ag_out['age']
                case_feature['gender'] = ag_out['gender']
                case_feature['start'] = sent_ind
                case_feature['link'] = ll_artl["link"]
                num_cases.append(case_feature)
        # Identifying the start sentence and end sentence for each case
        for case_ind in xrange(len(num_cases)):
            sent_start = num_cases[case_ind]['start']
            sent_end = sent_start + 2
            for ind_case in xrange(len(num_cases)):
                if ind_case == case_ind:
                    continue
                if num_cases[ind_case]['start'] > sent_start:
                    sent_end = num_cases[ind_case]['start'] - 1
                    break
            try:
                for ll_feat in seed_keywords:
                    num_cases[case_ind][ll_feat] = defaultdict()
                # Extracting the disease onset features for each case
                for dt_feat in ["Onset Date", "Hospital Date", "Outcome Date"]:
                    kwargs = {'K': K, 'w2v': w2v_model, 'start': sent_start, 'end': sent_end, 
                              'll_sents': ll_sents, 'seed': seed_keywords[dt_feat], 'dt_dict': dt_dict}
                    num_cases[case_ind][dt_feat] = ll_extract.infer_date(**kwargs)['final']
                # Extracting the clinical features corresponding to each case
                for clin_feat in ["Specified Proximity to Animals or Animal Products", 
                                  "Specified Contact with Other Cases", 
                                  "Specified HCW", "Specified Comorbidities"]:
                    kwargs = {'K': K, 'w2v': w2v_model, 'start': sent_start, 'end': sent_end, 
                              'll_sents': ll_sents, 'seed': seed_keywords[clin_feat]}
                    num_cases[case_ind][clin_feat] = ll_extract.infer_clinical(**kwargs)['final']
            except Exception as e:
		        print e
        if len(num_cases) != 0:
            auto_ll.extend(num_cases)
    # Writing the automatically extracted line lists to a file
    if len(auto_ll) != 0:
        with open(_arg.outputll, "w") as f_ll:
            for cs in auto_ll:
                print >> f_ll, json.dumps(cs, encoding='utf-8')


if __name__ == "__main__":
    main()
