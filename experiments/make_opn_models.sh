#!/bin/bash
python run_dense_index_experiment.py --index="qwen_4b.index" --model="Qwen/Qwen3-Embedding-4B" --padding_left=True
python run_dense_index_experiment.py --index="qwen_06b.index" --model="Qwen/Qwen3-Embedding-0.6B" --padding_left=True
python run_dense_index_experiment.py --index="qwen_8b.index" --model="Qwen/Qwen3-Embedding-8B" --padding_left=True
python run_dense_index_experiment.py --index="gte-Qwen2_1-5b.index" --model="Alibaba-NLP/gte-Qwen2-1.5B-instruct" --padding_left=False
python run_dense_index_experiment.py --index="gte-Qwen2_7b.index" --model="Alibaba-NLP/gte-Qwen2-7B-instruct" --padding_left=False
python run_dense_index_experiment.py --index="inf_retriever_v1_small.index" --model="infly/inf-retriever-v1-1.5b" --padding_left=False
python run_dense_index_experiment.py --index="inf_retriever_v1.index" --model="infly/inf-retriever-v1" --padding_left=False
python run_dense_index_experiment.py --index="sfr_2r_7b.index" --model="Salesforce/SFR-Embedding-2_R" --padding_left=False
python run_dense_index_experiment.py --index="sfr_mistral_7b.index" --model="Salesforce/SFR-Embedding-Mistral" --padding_left=False
python run_dense_index_experiment.py --index="linq_mistral.index" --model="Linq-AI-Research/Linq-Embed-Mistral" --padding_left=False
python run_dense_index_experiment.py --index="mistral_7b.index" --model="intfloat/e5-mistral-7b-instruct" --padding_left=False
python run_dense_index_experiment.py --index="jina_v4_symmetric.index" --model="jinaai/jina-embeddings-v4" --padding_left=False

