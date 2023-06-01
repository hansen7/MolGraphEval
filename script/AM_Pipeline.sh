#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainData_List=(geom2d_nmol500000_nconf1)
export PreTrainer=AM
# export MaskRate_List=($(seq 0.05 0.05 0.96))
export MaskRate_List=($(seq 0.1 0.1 0.96))
export LearningRate_List=(0.01 0.005 0.001 0.0005 0.0001)
# export PreTrainSeed_List=(1 2 3)

# for PreTrainSeed in "${PreTrainSeed_List[@]}"; do 
export PreTrainSeed=1
for PreTrainData in "${PreTrainData_List[@]}"; do
for MaskRate in "${MaskRate_List[@]}"; do
for LearningRate in "${LearningRate_List[@]}"; do 
    python src/run_pretraining.py \
        --mask_rate="$MaskRate" \
        --pretrainer="$PreTrainer" \
        --dataset="$PreTrainData" \
        --seed="$PreTrainSeed" \
        --lr="$LearningRate" \
        --epochs=100 \
        --output_model_dir=./pretrain_models_hyperopt/;
done
done
done

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export Seed_List=(1 2 3)

for MaskRate in "${MaskRate_List[@]}"; do
for LearningRate in "${LearningRate_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/mask_rate-${MaskRate}_seed-${PreTrainSeed}_lr-${LearningRate}"
    echo $Checkpoint_Folder
    # === Embedding Extractions ===
    python src/run_embedding_extraction.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --input_model_file=./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth \
        --embedding_dir=./embedding_dir_x/"$Checkpoint_Folder"/;
    # === Probing ===
    python src/run_validation.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --probe_task="$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done >> ${PreTrainer}_${PreTrainData}_mask_rate-${MaskRate}_seed-${PreTrainSeed}_lr-${LearningRate}.log
done
done

for MaskRate in "${MaskRate_List[@]}"; do
for LearningRate in "${LearningRate_List[@]}"; do
for Seed in "${Seed_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/mask_rate-${MaskRate}_seed-${Seed}_lr-${LearningRate}"
    echo $Checkpoint_Folder
    # === Fine-Tuning the Models ===
    python src/run_validation.py \
        --val_task="finetune" \
        --seed="${Seed}" \
        --dataset="$FineTuneData" \
        --pretrainer="FineTune" \
        --probe_task="downstream" \
        --input_model_file=./pretrain_models_hyperopt/"$Checkpoint_Folder"/epoch99_model_complete.pth ;
done >> ${PreTrainer}_${PreTrainData}_mask_rate-${MaskRate}_seed-${Seed}_lr-${LearningRate}_FineTune.log
done
done
done