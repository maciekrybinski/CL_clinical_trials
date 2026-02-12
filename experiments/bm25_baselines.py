#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 19:19:22 2025

@author: maciek
"""


import re
import csv
import xml
import subprocess
import ir_datasets

dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
queries = list(dataset.queries_iter())

def save_topics_as_csv(queries, out):
    with open(out, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for topic in queries:
            writer.writerow([topic['id'], re.sub(r'[^a-zA-Z0-9\s]', '', topic['text']).replace('\n', ' ')])

def parse_topics(f): 
    topicList = []
    #print (str(f))
    DOMTree = xml.dom.minidom.parse(f)
    collection = DOMTree.documentElement
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        topicDict = dict()
        
        topicDict["id"] = topic.getAttribute("number")
        topicDict["text"]=topic.firstChild.nodeValue.strip()
        topicList.append(topicDict)
    return topicList

topic_files=['../topics_src_to_eng/topics2021.xml',
             '../topics_src_to_eng/topics_from_spa_Latn.xml',
             '../topics_src_to_eng/topics_from_ita_Latn.xml',
             '../topics_src_to_eng/topics_from_pol_Latn.xml',
             '../topics_src_to_eng/topics_from_tur_Latn.xml',
             '../topics_src_to_eng/topics_from_ben_Beng.xml',
             '../topics_src_to_eng/topics_from_ell_Grek.xml',
             '../topics_src_to_eng/topics_from_eus_Latn.xml']
results=[['Lang', 'nDCG@10', 'P@10', 'RR']]
for f in topic_files:
    topics=parse_topics(f)
    if len(f.split('from_'))==1:
        out='default.tsv'
    else:
        out=f.split('from_')[1]+'.tsv'
    save_topics_as_csv(topics, out)
    subprocess.run(["python", "-m", "pyserini.search.lucene", "--index", "../indices/ct2021",
                    "--topics", out, "--output", 'bm25'+out.replace('tsv','txt'),
                    "--bm25"])
    import ir_measures
    from ir_measures import P, RR, NDCG
    qrels = dataset.qrels
    qrels_b = ir_measures.read_trec_qrels('../qrels2021_binary.txt')
    run = list(ir_measures.read_trec_run('bm25'+out.replace('tsv','txt')))
    result=ir_measures.calc_aggregate([NDCG@10], qrels, run)
    result={str(k):result[k] for k in result}
    result_b=ir_measures.calc_aggregate([RR,  P@10], qrels_b, run)
    result_b={str(k):result_b[k] for k in result_b}
    results.append([out.replace('.tsv', ''), round(result['nDCG@10'],2), round(result_b['RR'],2),round(result_b['P@10'],2)])


