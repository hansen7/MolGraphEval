#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainer=AM

export PreTrainSeed=1
export MaskRate=0.85  # 0.50
export LearningRate=0.0001  # 0.0005
export PreTrainData=geom2d_nmol500000_nconf1
export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export ProbeTask_List=(node_degree node_centrality node_clustering link_prediction jaccard_coefficient katz_index \
    graph_diameter node_connectivity cycle_basis assortativity_coefficient average_clustering_coefficient)

mkdir -p log/${PreTrainer}/${PreTrainer}_hyperopt_metric/
export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/mask_rate-${MaskRate}_seed-${PreTrainSeed}_lr-${LearningRate}"
echo $Checkpoint_Folder

# === Embedding Extractions ===
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_embedding_extraction.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --input_model_file=./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth \
        --embedding_dir=./embedding_dir_x/"$Checkpoint_Folder"/;
done

for ProbeTask in "${ProbeTask_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --probe_task="$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done
done >> log/${PreTrainer}/${PreTrainer}_hyperopt_metric/_Metric_Probe.log