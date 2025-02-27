# Command Line Interface

## Evaluating checkpoints of K-fold cross-validation

### Lung Cancer Subtype Classification (TCGA-NSCLC)

```bash
python3 evaluator.py \
        --seed 2024 \
        --k_start 0\
        --k_end 4\
        --input_size 1024\
        --config '../yaml/tcga_lung.yml'\
        --checkpoint_dir '../weights/nsclc'\
        --splits_dir '../splits/nsclc/'\
        --ratio_graph '0.2'\
        --alignment '../weights/alignment/plip_alignment.pth'\
        --free_text_encoder \
        --output_dir 'eval_results'
```
