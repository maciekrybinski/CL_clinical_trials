import ir_datasets
import json

dataset = ir_datasets.load("clinicaltrials/2021")

i=0
for doc in dataset.docs_iter():
    # print(doc.doc_id)
    content = ' '.join([doc.title, doc.summary, doc.detailed_description, doc.eligibility])
    content=content.replace('\n',' ').replace('\r', '')
    i+=1
    if i%1000==0:
        print(i)
    data={"id":doc.doc_id, "contents":content}
    with open('./raw_ct_data/'+doc.doc_id+'.json', 'w') as f:
        json.dump(data, f)
print (i)    

"""
python -m pyserini.index.lucene   --collection JsonCollection   --input raw_ct_data   --index ../indices/ct2021   --generator DefaultLuceneDocumentGenerator   --threads 1   --storePositions --storeDocvectors --storeRaw
"""
