#!/bin/bash
# shellcheck disable=SC2164
cd ../src

export PreTrainData=geom2d_nmol50000_nconf1_nupper1000
export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export PreTrainer=Contextual
export ProbeTask=downstream

for FineTuneData in "${FineTuneData_List[@]}"; do
    echo
    echo
    echo $FineTuneData
    python run_validation.py \
        --pretrainer="$PreTrainer" \
        --dataset="$FineTuneData" \
        --probe_task="$ProbeTask" \
        --embedding_dir=./embedding_dir_x/${PreTrainer}/${PreTrainData}/;
done >> ../script/${PreTrainer}_${PreTrainData}.log
