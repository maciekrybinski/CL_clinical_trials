#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:01:03 2025

@author: maciek
"""

import ir_datasets
import json

import copy 
import xml.dom.minidom
import argparse
import sys
sys.path.append('../indexing/')
from persistent_index import FaissManager, QwenVectoriser, JinaVectoriser

def parse_topics(f): 
    topicList = []
    DOMTree = xml.dom.minidom.parse(f)
    collection = DOMTree.documentElement
    topics = collection.getElementsByTagName("topic")
    for topic in topics:
        topicDict = dict()
        
        topicDict["id"] = topic.getAttribute("number")
        topicDict["text"]=topic.firstChild.nodeValue.strip()
        
        topicList.append(topicDict)

    return topicList

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--index', action="store", dest='index')
parser.add_argument('--model', action="store", dest='model')
parser.add_argument('--prompt', action="store", dest='prompt', default='Given a patient note, find clinical trials the patient is eligible for.')
parser.add_argument('--padding_left', action="store", dest='padding_left', default=False)
parser.add_argument('--trust_remote', action="store", dest='trust_remote', default=True)
parser.add_argument('--output', action="store", dest='output', default='output.txt')
parser.add_argument('--topics', action="store", dest='topics', 
                    default='topics2021_en.xml;topics2021_es.xml;topics2021_it.xml;topics2021_pl.xml;topics2021_tr.xml;topics2021_el.xml;topics2021_eu.xml;topics2021_bn.xml')
args = parser.parse_args()

models=["Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "infly/inf-retriever-v1", "infly/inf-retriever-v1-1.5b", "jinaai/jina-embeddings-v4", 
        "Linq-AI-Research/Linq-Embed-Mistral", "Salesforce/SFR-Embedding-2_R", 
        "Salesforce/SFR-Embedding-Mistral", "intfloat/e5-mistral-7b-instruct"]
if args.model not in models:
    raise Exception('Model not supported.')

dataset = ir_datasets.load("clinicaltrials/2021")
docs=[]
doc_id_index={}
for doc in dataset.docs_iter():
    content = ' '.join([doc.title, doc.summary, doc.eligibility])
    docs.append(content)
    doc_id_index[len(docs)-1]=doc.doc_id
    


m=FaissManager()
if 'jina' not in args.model:
    m.vectoriser=QwenVectoriser(model=args.model, left_side_padding=args.padding_left,
                                trust_remote=args.trust_remote)
else:
    m.vectoriser=JinaVectoriser(model=args.model)
m.load_index('../indices/'+args.index)

topic_files= [f.strip() for f in args.topics.split(';')]
for tf in topic_files:
    print(tf)
    results=[]
    ts=parse_topics('../topics/'+tf)
    for topic in ts:
        di2, ne2 = m.text_search(topic["text"], 
                                 prompt=args.prompt, k=1000)
        r=[[topic['id'], 'Q0', doc_id_index[x[0]], (i+1), (1-x[1]), args.model] for (i, x) in enumerate(zip(ne2[0], di2[0]))]
        results.extend(r)
        
    import csv
    with open('../results/'+args.model.replace('/','-')+'_'+tf.replace('.xml','').replace('/','_')+'.txt', 'w', newline='\n') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for line in results:
            tsv_output.writerow(line)
