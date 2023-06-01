#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainData=zinc_standard_agent
export LearningRate=0.001
export PreTrainer=RGCL
export PreTrainSeed=1

python src/run_pretraining.py \
    --project_name="$PreTrainer" \
    --pretrainer="$PreTrainer" \
    --dataset="$PreTrainData" \
    --seed="$PreTrainSeed" \
    --lr="$LearningRate" \
    --output_model_dir="./pretrain_models_hyperopt/";

export ProbeTask=downstream
export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)

mkdir -p log/${PreTrainer}_ZINC/

# == Linear Probing ==
for FineTuneData in "${FineTuneData_List[@]}"; do
    export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/_seed-${PreTrainSeed}_lr-${LearningRate}"
    echo $Checkpoint_Folder
    # === Embedding Extractions ===
    python src/run_embedding_extraction.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --input_model_file=./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth \
        --embedding_dir=./embedding_dir_x/"$Checkpoint_Folder"/;
    # === Fine-Tuning the Models ===
    python src/run_validation.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --probe_task="$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done >> log/${PreTrainer}_ZINC/${PreTrainData}_seed-${PreTrainSeed}_lr-${LearningRate}.log

# == End2End Fine-Tuning ==
export TrainSeed_List=(1 2 3)
for Seed in "${TrainSeed_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/_seed-${PreTrainSeed}_lr-${LearningRate}"
    echo $Checkpoint_Folder
    # === Fine-Tuning the Models ===
    python src/run_validation.py \
        --val_task="finetune" \
        --seed="${Seed}" \
        --dataset="$FineTuneData" \
        --pretrainer="FineTune" \
        --probe_task="$ProbeTask" \
        --input_model_file=./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth ;
done >> log/${PreTrainer}_ZINC/${PreTrainData}_seed-${PreTrainSeed}_lr-${LearningRate}_FineTune.log
done
