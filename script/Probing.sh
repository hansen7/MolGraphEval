#!/bin/bash
cd ../

export PreTrainer=AM # or others
export PreTrainData=zinc_standard_agent # or others
export EmbeddingDir="./embedding_dir/$PreTrainer/$PreTrainData/"
export Checkpoint="./pretrain_models/$PreTrainer/$PreTrainData/epoch99_model_complete.pth"
# for AM, the default path of checkpoint is "./pretrain_models/$PreTrainer/$PreTrainData/mask_rate-0.15_seed-42_lr-0.001/epoch0_model_complete.pth"
# for the random baseline, just set the checkpoint as epoch0_model_complete.pth

export FineTuneData_List=(bbbp tox21 toxcast sider clintox muv hiv bace)

# === Embedding Extractions, from Fixed Pre-Trained GNNs ===
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_embedding_extraction.py \
        --dataset="$FineTuneData" \
        --pretrainer="$PreTrainer" \
        --embedding_dir="$EmbeddingDir" \
        --input_model_file="$Checkpoint" ;
done

# === Probing on Downstream Tasks ===
export ProbeTask=downstream
for FineTuneData in "${FineTuneData_List[@]}"; do
    python run_validation.py \
        --dataset="$FineTuneData" \
        --probe_task="$ProbeTask" \
        --pretrainer="$PreTrainer" \
        --embedding_dir="$EmbeddingDir";
done >> Downstream_Probe_${PreTrainer}_${PreTrainData}.log


# === Probing on Topological Metrics ===
export ProbeTask=node_degree # or node_centrality node_clustering link_prediction jaccard_coefficient katz_index graph_diameter node_connectivity cycle_basis assortativity_coefficient average_clustering_coefficient
for FineTuneData in "${FineTuneData_List[@]}"; do
    python src/run_validation.py \
        --probe_task="$ProbeTask" \
        --dataset="$FineTuneData" \
        --pretrainer="$PreTrainer" \
        --embedding_dir="$EmbeddingDir" ;
done >> Topological_Probe_${PreTrainer}_${PreTrainData}.log


# === Probing on Substructures ===
export Substructure_List=(fr_epoxide fr_lactam fr_morpholine fr_oxazole \
fr_tetrazole fr_N_O fr_ether fr_furan fr_guanido fr_halogen fr_morpholine \
fr_piperdine fr_thiazole fr_thiophene fr_urea fr_allylic_oxid fr_amide \
fr_amidine fr_azo fr_benzene fr_imidazole fr_imide fr_piperzine fr_pyridine)

for Substructure in "${Substructure_List[@]}"; do
    for FineTuneData in "${FineTuneData_List[@]}"; do
        python src/run_validation.py \
            --dataset="$FineTuneData" \
            --pretrainer="$PreTrainer" \
            --embedding_dir="$EmbeddingDir" \
            --probe_task="RDKiTFragment_$Substructure" ;
    done
done >> Substructure_Probe_${PreTrainer}_${PreTrainData}.log
