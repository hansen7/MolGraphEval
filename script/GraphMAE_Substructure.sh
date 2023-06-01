#!/bin/bash
cd ~/GraphEval_dev

export PreTrainData=zinc_standard_agent
export LearningRate=0.001
export PreTrainer=GraphMAE
export PreTrainSeed=1

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export Substructure_List=(fr_epoxide fr_lactam fr_morpholine fr_oxazole \
fr_tetrazole fr_N_O fr_ether fr_furan fr_guanido fr_halogen fr_morpholine \
fr_piperdine fr_thiazole fr_thiophene fr_urea fr_allylic_oxid fr_amide \
fr_amidine fr_azo fr_benzene fr_imidazole fr_imide fr_piperzine fr_pyridine)
export Checkpoint_Folder="${PreTrainer}/${PreTrainData}/_seed-${PreTrainSeed}_lr-${LearningRate}"

mkdir -p log/${PreTrainer}_ZINC/

for ProbeTask in "${Substructure_List[@]}"; do
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --pretrainer=$PreTrainer \
        --dataset=$FineTuneData \
        --probe_task="RDKiTFragment_$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${Checkpoint_Folder}/;
done
done >> log/${PreTrainer}_ZINC/_Substructure_Probe.log