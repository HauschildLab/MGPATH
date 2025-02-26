# Command Line Interface

## Evaluating checkpoints of K-fold cross-validation

### Lung Cancer Subtype Classification (TCGA-NSCLC)

```bash
python3 evaluator.py \
    --k_start 0 \
    --k_end 5 \
    --config yaml/tcga_lung.yml \
    --checkpoint_dir checkpoints/tcga_lung
    --splits_dir splits/tcga_lung
    --output_dir eval_results/tcga_lung
```
