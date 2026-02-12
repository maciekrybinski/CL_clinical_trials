Intro

Welcome to the GitHub repository for our paper 'CT_CL: A Cross-Language Benchmark for Matching Patients to Clinical Trials'.

The success of clinical trials depends on the recruitment of patients who match strict inclusion criteria. The development of effective patient to clinical trial matching systems depends on benchmarking datasets that support systematic evaluation. Apart from resources that are created in English by the TREC Clinical trials tracks (2021-23), very limited corpora exist for other languages and cross-language settings, despite the need of automatic support for clinical trial recruitment being global. To address this gap, we combine machine translation with medical expert annotation to construct CT$_{CL}$ (Clinical Trials Cross Lingual retrieval), a cross-lingual evaluation benchmark for patient-clinical trial retrieval in seven languages. We benchmark the cross-lingual retrieval task using 14 large language (embedding) models. We showcase how our dataset can be used to evaluate the cross-lingual capability of the models for languages with varying resource availability.

The topics translated and validated in seven languages (Basque, Bengali, Greek, Italian, Polish, Spanish, Turkish; we also include English for convenience) can be found in the 'topics' subfolder. Each file ends with a corresponding two-letter language code. A persistent data repository for the project (in particular, the data files of our dataset) can be found at XXX.

The repository contains a simple quicksart how-to example showing how topics for each of the languages can be read into a Python structure. More comprehensive examples (notebooks) can be found in the 'examples' subfolder. Full code to reproduce our experiments from the paper is in the 'experiments' subfolder. Running the full experiments will require running the indexing jobs first (see folder 'indexing'), to create persistent indices in the 'indices' subfolder. We attach a requirements file, which should help with setting up the environment for running the experiments.

Quickstart
Translated topics can be read in Python, the same way one would read the original topics of the TREC CT 2021 collection. They are formatted with XML. If you check out the repository, elow example code snippet executed from the root folder reads the XML and writes a tsv queries stripped from interpunction, which is the format to use in an experiment with (for example) Pyserini BoW statistical models. 


import csv
import xml
def save_topics_as_csv(queries, out):
    with open(out, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        for topic in queries:
            writer.writerow([topic['id'], re.sub(r'[^a-zA-Z0-9\s]', '', topic['text']).replace('\n', ' ')])

def read_topics(f): 
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
# read Spanish topics - ES language code
t=read_topics('./topics/topics2021_es.xml')
# save spanish topics line by line
save_topics_as_csv(t, 'out_es.tsv')

Key dependencies and requirements.txt
Key dependencies for our experimental code include: pytorch, transformers, sentence transformers, faiss, pyserini, panda, numpy, ir_datasets, and nltk. We include the requirements.txt if you wish to replicate our python environment.

Evaluating runs
Our experimental code generates TREC-formatted runs. If you like to evaluate the runs we recommend using ir_measures package. For convenience, we include human judgements (identical to those used in TREC CT 2021) in the 'qrels' subfolder of this repository.

