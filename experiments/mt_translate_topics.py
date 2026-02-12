#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 10:13:21 2026

@author: maciek
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk import sent_tokenize
import xml
import xml.etree.cElementTree as ET
import timeit


def save_topics(topics_list, out):
    topics=ET.Element("topics", task="2021 TREC Clinical Trials")
    i=1
    for t in topics_list:
        # print(str(t))
          topic=ET.SubElement(topics, "topic", number=str(i)).text=t
    
          i+=1
    tree = ET.ElementTree(topics)
    tree.write(out)
    
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


def translate_text(text, src, trg):
    MODEL = "facebook/nllb-200-3.3B"
    # src='ben_Beng'
    # trg='eng_Latn'



    print(f'Using model: {MODEL}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                              src_lang=src)  # BCP-47 code
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).half().cuda()
    print('Ok')
    start = timeit.timeit()
    if src=='ben_Beng':
        sentences = text.split('।')
        segments = [s.strip() for s in sentences if s.strip()]
    elif src=='spa_Latn':
        segments = sent_tokenize(text, 'spanish')
    elif src=='ell_Grek':
        segments = sent_tokenize(text, 'greek')
    elif src=='ita_Latn':
        segments = sent_tokenize(text, 'italian')
    elif src=='pol_Latn':
        segments = sent_tokenize(text, 'polish')
    elif src=='tur_Latn':
        segments = sent_tokenize(text, 'turkish')
    elif src=='eus_Latn':
        segments = sent_tokenize(text, 'spanish')

    translated = []
    
    for idx, segment in enumerate(segments):
        if not len(segment):
            decoded = segment
        else:

            inputs = tokenizer(segment, return_tensors="pt").to('cuda')


            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(trg),
                max_length=1024
            )
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        translated.append(decoded)
        print(decoded)
    end = timeit.timeit()
    return ' '.join(translated), end - start

# source_files={'./topics/topics2021_validated_es.xml':'spa_Latn'}

source_files={'../topics/topics2021_es.xml':'spa_Latn',
              '../topics/topics2021_el.xml':'ell_Grek',
              '../topics/topics2021_it.xml':'ita_Latn',
              '../topics/topics2021_pl.xml':'pol_Latn',
              '../topics/topics2021_tr.xml':'tur_Latn', 
              '../topics/topics2021_bn.xml':'ben_Beng',
              '../topics/topics2021_eu.xml':'eus_Latn'}
elapsed_times=[]
for file in source_files:
    topics_dict_src=parse_topics(file)
    translated_topics=[]
    for t in topics_dict_src:
        print(t['id'])
        print(source_files[file])
        translated, elapsed=translate_text(t['text'], source_files[file], 'eng_Latn')
        translated_topics.append(translated)
        elapsed_times.append(elapsed)
    
    save_topics(translated_topics, 'topics_from_'+source_files[file]+'.xml')

