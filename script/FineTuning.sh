#!/bin/bash
cd ../

export PreTrainer=AM # or others
export PreTrainData=zinc_standard_agent # or others

export ProbeTask=downstream
export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)
export Checkpoint="./pretrain_models/$PreTrainer/$PreTrainData/epoch99_model_complete.pth"

for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --val_task="finetune" \
        --pretrainer="FineTune" \
        --dataset=$FineTuneData \
        --probe_task=$ProbeTask \
        --input_model_file=$Checkpoint ;
done >> FineTune_${PreTrainer}_${PreTrainData}.log
