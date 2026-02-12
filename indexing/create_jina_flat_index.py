#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:01:03 2025

@author: maciek
"""

import ir_datasets
import json

dataset = ir_datasets.load("clinicaltrials/2021")

docs=[]
doc_id_index={}
for doc in dataset.docs_iter():
    content = ' '.join([doc.title, doc.summary, doc.eligibility])
    docs.append(content)
    doc_id_index[len(docs)-1]=doc.doc_id
    
from persistent_index import FaissManager, JinaVectoriser
m=FaissManager()
m.vectoriser=JinaVectoriser()
m.create_index_from_strings(docs, list(range(len(docs))))
m.save_index('../indices/jina_v4_symmetric.index')
