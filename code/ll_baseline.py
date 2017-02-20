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
import sys
import networkx as nx
import numpy as np
import nltk

regex1 = re.compile(r'.*\s+(?P<age>\d{1,2})(.{0,20})(\s+|-)(?P<gender>woman|man|male|female|boy|girl|housewife).*')
regex2 = re.compile(r'.*\s+(?P<age>\d{1,2})\s*years?(\s|-)old.*')
regex3 = re.compile(r'.*\s*(?P<gender>woman|man|male|female|boy|girl|housewife|he|she).*')
negation_cues = ["no", "not", "none", "cannot", "without", "lack", "absence"]
_digits = re.compile('\d')


def contains_digits(word):
    return bool(_digits.search(word))


def get_age_gender(t):
        
    m1 = regex1.match(t)
    if m1:
        age = int(m1.groupdict()['age'])
        gender = m1.groupdict()['gender']
    else:
        m2 = regex2.match(t)
        if m2:
            age = int(m2.groupdict()['age'])
        else:
            age = None
                                                                                                        
        m3 = regex3.match(t)
        if m3:
            gender = m3.groupdict()['gender']
        else:
            gender = None
    return {'age': age, 'gender': get_gender_value(gender)}


def get_gender_value(x):
    
    if x:
        x = x.strip(r'\s*|-')                    
        if x in ['man', 'male', 'boy', 'he']:
            return 'M'
        elif x in ['woman', 'female', 'girl', 'housewife', 'she']:
            return 'F'
    return None


def infer_onset(sent_start, sent_end, ll_sents, seed_kw, dt_dict):
    
    predictors = []
    predictor_dist = defaultdict()
    predictors.append(seed_kw)
    for predictor in predictors:
        predictor_dist[predictor] = defaultdict()
        for dt_str in dt_dict:
            predictor_dist[predictor][dt_str] = []
    sent_start_iter = sent_start
    while sent_start_iter <= sent_end:
        sent_dg = nx.Graph()
        for token in ll_sents[sent_start_iter]:
            if token.head.orth_ != token.orth_:
                sent_dg.add_edge(token.head, token)
        for predictor in predictor_dist:
            for dt_str in dt_dict:
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
                        predictor_dist[predictor][dt_str].append(nx.shortest_path_length(sent_dg, pred_obj, dt_obj))
                except Exception:
                    continue
        sent_start_iter += 1
    predictor_forecast = {}
    for predictor in predictor_dist:
        predictor_forecast[predictor] = ""
        for dt_str in dt_dict:
            if len(predictor_dist[predictor][dt_str]) == 0:
                predictor_dist[predictor].pop(dt_str)
            else:
                predictor_dist[predictor][dt_str] = np.min(predictor_dist[predictor][dt_str])
        if len(predictor_dist[predictor]) > 0:
            predictor_forecast[predictor] = min(predictor_dist[predictor].items(), key=lambda x: x[1])[0]
        if predictor_forecast[predictor] == "":
            predictor_forecast.pop(predictor)
    try:
        final_forecast = dt_dict[nltk.FreqDist(predictor_forecast.values()).max()]
    except Exception:
        final_forecast = None
    return final_forecast


def infer_clinical(sent_start, sent_end, ll_sents, seed_kw):
    
    predictors = []
    predictor_forecast = defaultdict()
    predictors.append(seed_kw)
    for predictor in predictors:
        sent_start_iter = sent_start
        while sent_start_iter <= sent_end:
            is_detect = 0
            sent_dg = nx.DiGraph()
            for token in ll_sents[sent_start_iter]:
                if token.head.orth_ != token.orth_:
                    sent_dg.add_edges_from([(token.head, token)])
            for nd in sent_dg.nodes():
                if nd.orth_ == predictor:
                    is_detect = 1
                    is_negation = 0
                    nd_neighbors = list(set(sent_dg.neighbors(nd)))
                    for neigh_elm in nd_neighbors:
                        if neigh_elm.orth_ in negation_cues:
                            is_negation = 1
                    if not is_negation:
                        for nd_elm in sent_dg.nodes():
                            try:
                                if nx.shortest_path(sent_dg, nd_elm, nd):
                                    nd_elm_neighbors = list(set(sent_dg.neighbors(nd_elm)))
                                    for neigh_elm in nd_elm_neighbors:
                                        if neigh_elm.orth_ in negation_cues:
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
    if len(predictor_forecast) == 0:
        final_forecast = 'N'
    else:
        final_forecast = nltk.FreqDist(predictor_forecast.values()).max()
    return final_forecast


def main():

    ll_articles = [json.loads(l) for l in io.open(sys.argv[1], "r")]
    seed_keywords = {"Onset Date": "onset", 
                     "Hospital Date": "hospitalized", 
                     "Outcome Date": "died", 
                     "Specified Proximity to Animals or Animal Products": "animals", 
                     "Specified Contact with Other Cases": "case", 
                     "Specified HCW": "healthcare",
                     "Specified Comorbidities": "comorbidities"}
    auto_ll = []
    for ll_artl in ll_articles:
        ll_text = ""
        dt_offsets = []
        dt_dict = defaultdict()
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
        for sent_ind in xrange(len(ll_sents)):
            ag_out = get_age_gender(ll_sents[sent_ind].text)
            if ag_out['age'] is not None and ag_out['gender'] is not None:
                case_feature = defaultdict()
                case_feature['age'] = ag_out['age']
                case_feature['gender'] = ag_out['gender']
                case_feature['start'] = sent_ind
                case_feature['link'] = ll_artl["link"]
                num_cases.append(case_feature)
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
                for seed_key in seed_keywords:
                    num_cases[case_ind][seed_key] = defaultdict()
                num_cases[case_ind]["Onset Date"] = infer_onset(sent_start, sent_end, 
                                                                ll_sents, seed_keywords['Onset Date'], 
                                                                dt_dict)
                num_cases[case_ind]["Hospital Date"] = infer_onset(sent_start, sent_end, 
                                                                   ll_sents, seed_keywords['Hospital Date'], 
                                                                   dt_dict)
                num_cases[case_ind]["Outcome Date"] = infer_onset(sent_start, sent_end, 
                                                                  ll_sents, seed_keywords['Outcome Date'], 
                                                                  dt_dict)
                for clin_feat in ["Specified Proximity to Animals or Animal Products", "Specified Contact with Other Cases", 
                                  "Specified HCW", "Specified Comorbidities", "Potentially Nosocomial"]:
                    num_cases[case_ind][clin_feat] = infer_clinical(sent_start, sent_end, ll_sents, seed_keywords[clin_feat])
            except Exception:
                continue
        if len(num_cases) != 0:
            auto_ll.extend(num_cases)
    if len(auto_ll) != 0:
        with open(sys.argv[2], "w") as f_ll:
            for cs in auto_ll:
                print >> f_ll, json.dumps(cs, encoding='utf-8')


if __name__ == "__main__":
    main()
