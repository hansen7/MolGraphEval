#!/bin/bash
cd ~/GraphEval_dev

export PreTrainData=zinc_standard_agent
export LearningRate=0.001
export PreTrainer=GraphMAE
export PreTrainSeed=1

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export ProbeTask_List=(node_degree node_centrality node_clustering link_prediction jaccard_coefficient katz_index \
    graph_diameter node_connectivity cycle_basis assortativity_coefficient average_clustering_coefficient)
export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/_seed-${PreTrainSeed}_lr-${LearningRate}"

mkdir -p log/${PreTrainer}_ZINC/

for ProbeTask in "${ProbeTask_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --pretrainer=$PreTrainer \
        --dataset=$FineTuneData \
        --probe_task=$ProbeTask \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done
done >> log/${PreTrainer}_ZINC/_Metric.log
