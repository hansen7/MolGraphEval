#!/bin/bash
# shellcheck disable=SC2164
cd ~/GraphEval_dev

export PreTrainData=zinc_standard_agent
export PreTrainer=GraphCL
export LearningRate=0.001
export Aug_Strength=0.2
export PreTrainSeed=1
export Aug_Prob=0.1

mkdir -p log/${PreTrainer}_ZINC/

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export Substructure_List=(fr_epoxide fr_lactam fr_morpholine fr_oxazole \
fr_tetrazole fr_N_O fr_ether fr_furan fr_guanido fr_halogen fr_morpholine \
fr_piperdine fr_thiazole fr_thiophene fr_urea fr_allylic_oxid fr_amide \
fr_amidine fr_azo fr_benzene fr_imidazole fr_imide fr_piperzine fr_pyridine)

export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/aug_mode-sample_aug_strength-${Aug_Strength}_aug_prob-${Aug_Prob}_seed-${PreTrainSeed}_lr-${LearningRate}"
echo $Checkpoint_Folder


for ProbeTask in "${Substructure_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --pretrainer=$PreTrainer \
        --dataset=$FineTuneData \
        --probe_task="RDKiTFragment_$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done
done >> log/${PreTrainer}_ZINC/_Substructure_Probe.log