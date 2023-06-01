#!/bin/bash
# shellcheck disable=SC2164
cd ../src

export PreTrainData_List=(geom2d_nmol50000_nconf1_nupper1000 geom2d_nmol100000_nconf1_nupper1000 \
    geom2d_nmol200000_nconf1_nupper1000 geom2d_nmol500000_nconf1_nupper1000)
export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export PreTrainer=AM

# === Pre-Trainings ===

for PreTrainData in "${PreTrainData_List[@]}"; do
    python run_pretraining.py \
        --pretrainer="$PreTrainer" \
        --dataset="$PreTrainData" \
        --epochs=100 \
        --output_model_dir=./saved_models_x/;
done

# === Embedding Extractions ===

for PreTrainData in "${PreTrainData_List[@]}"; do
    for FineTuneData in "${FineTuneData_List[@]}"; do
        python run_embedding_extraction.py \
            --pretrainer="$PreTrainer" \
            --dataset="$FineTuneData" \
            --input_model_file=./saved_models_x/"$PreTrainer"/"$PreTrainData"/epoch99_model_complete.pth \
            --embedding_dir=./embedding_dir_x/"$PreTrainer"/"$PreTrainData"/;
    done
done

# === Probe Tasks ===

export ProbeTask_List=(node_degree node_centrality node_clustering \
    graph_diameter link_prediction jaccard_coefficient)

for PreTrainData in "${PreTrainData_List[@]}"; do
    for ProbeTask in "${ProbeTask_List[@]}"; do
        for FineTuneData in "${FineTuneData_List[@]}"; do
            python run_validation.py \
                --pretrainer="$PreTrainer" \
                --dataset="$FineTuneData" \
                --probe_task="$ProbeTask" \
                --embedding_dir=./embedding_dir_x/${PreTrainer}/${PreTrainData}/;
        done
    done >> ../script_hc/${PreTrainer}_${PreTrainData}.log
done
