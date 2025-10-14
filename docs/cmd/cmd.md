# Command Line Interface

## Training Models

### Lung Cancer Subtype Classification (TCGA-NSCLC)

```bash
typeGNN=gat_conv
ratio=0.2
fold=0
fold_next=5

export TOKENIZERS_PARALLELISM=false

python3 main.py \
        --seed 2023 \
        --drop_out \
        --early_stopping \
        --lr 9e-4 \
        --k 5 \
        --task 'task_tcga_lung_subtyping' \
        --results_dir </path/to/result/directory> \
        --exp_code <experiment codes> \
        --log_data \
        --data_folder_s </path/to/embedding/directory> \
        --data_folder_l </path/to/embedding/directory> \
        --data_graph_dir_s </path/to/embedding/directory>\
        --data_graph_dir_l </path/to/embedding/directory> \
        --aug_data_folder_s </path/to/embedding/directory>\
        --aug_data_folder_l </path/to/embedding/directory>\
        --aug_data_graph_dir_s </path/to/embedding/directory>\
        --aug_data_graph_dir_l </path/to/embedding/directory>\
        --split_dir 'few_shot_splits/TCGA-NSCLC/16shots/'  \
        --text_prompt_path 'tcga_nsclc.csv' \
        --max_epochs 50 \
        --typeGNN ${typeGNN} \
        --ratio_graph ${ratio} \
        --use_gigapath_backbone \
        --k_start ${fold} \
        --k_end ${fold_next} \
        --freeze_text_encoder
```

```bash
typeGNN=gat_conv
ratio=0.2
fold=0
fold_next=5

export TOKENIZERS_PARALLELISM=false

python3 main.py \
        --seed 2023 \
        --drop_out \
        --early_stopping \
        --lr 9e-4 \
        --k 5 \
        --task 'task_tcga_brca_subtyping' \
        --results_dir </path/to/result/directory> \
        --exp_code <experiment codes> \
        --log_data \
        --data_folder_s </path/to/embedding/directory> \
        --data_folder_l </path/to/embedding/directory> \
        --data_graph_dir_s </path/to/embedding/directory>\
        --data_graph_dir_l </path/to/embedding/directory> \
        --aug_data_folder_s </path/to/embedding/directory>\
        --aug_data_folder_l </path/to/embedding/directory>\
        --aug_data_graph_dir_s </path/to/embedding/directory>\
        --aug_data_graph_dir_l </path/to/embedding/directory>\
        --split_dir 'few_shot_splits/TCGA-NSCLC/16shots/'  \
        --text_prompt_path 'tcga_nsclc.csv' \
        --max_epochs 50 \
        --typeGNN ${typeGNN} \
        --ratio_graph ${ratio} \
        --use_gigapath_backbone \
        --k_start ${fold} \
        --k_end ${fold_next} \
        --freeze_text_encoder
```
