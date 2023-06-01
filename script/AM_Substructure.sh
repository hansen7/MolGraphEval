#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainer=AM

export PreTrainSeed=1
export MaskRate=0.85  # 0.50
export LearningRate=0.0001  # 0.0005
export PreTrainData=geom2d_nmol500000_nconf1
export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export Substructure_List=(fr_epoxide fr_lactam fr_morpholine fr_oxazole \
fr_tetrazole fr_N_O fr_ether fr_furan fr_guanido fr_halogen fr_morpholine \
fr_piperdine fr_thiazole fr_thiophene fr_urea fr_allylic_oxid fr_amide \
fr_amidine fr_azo fr_benzene fr_imidazole fr_imide fr_piperzine fr_pyridine)

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

for ProbeTask in "${Substructure_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --probe_task="RDKiTFragment_$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done
done >> log/${PreTrainer}/${PreTrainer}_hyperopt_metric/_Substructure_Probe.log
