#!/bin/bash
cd ../

export PreTrainData=zinc_standard_agent # or geom2d_nmol500000_nconf1
export PreTrainer=AM # or IM EP CP GPT_GNN JOAO JOAOv2 GraphCL Motif Contextual GraphMAE RGCL

# If the PreTrainer is GraphMVP, then the PreTrainData is geom3d_nmol500000_nconf1

# === Pre-Training ===
python src/run_pretraining.py \
    --dataset="$PreTrainData" \
    --pretrainer="$PreTrainer" \
    --output_model_dir=./pretrain_models/;
