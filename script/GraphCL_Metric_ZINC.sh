#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainData=zinc_standard_agent
export PreTrainer=GraphCL
export LearningRate=0.001
export Aug_Strength=0.2
export PreTrainSeed=1
export Aug_Prob=0.1

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export ProbeTask_List=(node_degree node_centrality node_clustering link_prediction jaccard_coefficient katz_index \
    graph_diameter node_connectivity cycle_basis assortativity_coefficient average_clustering_coefficient)

export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/aug_mode-sample_aug_strength-${Aug_Strength}_aug_prob-${Aug_Prob}_seed-${PreTrainSeed}_lr-${LearningRate}"
echo $Checkpoint_Folder
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
