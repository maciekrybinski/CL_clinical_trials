#!/bin/bash
python run_dense_index_experiment_nv.py --index="nv_v2.index" --model="nvidia/NV-Embed-v2" --padding_left=False
python run_dense_index_experiment_nv.py --index="nv_v1.index" --model="nvidia/NV-Embed-v1" --padding_left=False


