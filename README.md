# MGPATH
## MGPATH: Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot WSI Classification

Anh-Tien Nguyen, Duy Minh Ho Nguyen*, Nghiem Tuong Diep*, Trung Quoc Nguyen, Nhat Ho, Jacqueline Michelle Metsch, Miriam Cindy Maurer, Daniel Sonntag, Hanibal Bohnenberger, Anne-Christin Hauschild

(*Equal second contribution)

[![Static Badge](https://img.shields.io/badge/License-MGPath-brightgreen?link=https%3A%2F%2Fgithub.com%2FHauschildLab%2FMGPATH%2F)]()
[[`Model`](https://huggingface.co/tiennguyen/MGPATH/tree/main)] [[`Paper`](https://arxiv.org/abs/2502.07409)] [[`BibTeX`](#Citation)]


## 💥 📢 News 💥
- **[14.10.2025]**: Release all source codes !
- **[05.10.2025]**: [MGPATH] (https://openreview.net/forum?id=u7U81JLGjH) is published in Transactions on Machine Learning Research (TMLR) !
- **[11.02.2025]**: [MGPATH](https://arxiv.org/abs/2502.07409) is now available on arXiv !
- **[27.02.2025]**: MGPath(PLIP-G) [Hugging Face](https://huggingface.co/tiennguyen/MGPATH/tree/main) models are released !
- **[27.02.2025]**: PLIP-G aligment [Hugging Face](https://huggingface.co/tiennguyen/MGPATH/tree/main) models are released !
- **[27.02.2025]**: TCGA-NSCLC's embeddings [Hugging Face](https://huggingface.co/datasets/tiennguyen/MGPATH) extracted by Prov-GigaPath are release !

## Development Environment Installation

please refer to the installation guide [Development Environment Installation](docs/env/env.md)

## Reproducibility

For the testing, validation, and training slide IDs, please refer to the [splits](splits) directory.

## Model Download

The aligment (Prov-GigaPath and PLIP's text encoder) weight can be download from [Hugging Face](https://huggingface.co/tiennguyen/MGPATH/tree/main).
After downloading, please copy the weights to the directory [weights](weights) 

## Embedding Download

The embedding features of the mentioned datasets, extracted from GigaPath, can be downloaded from [Hugging Face](https://huggingface.co/datasets/tiennguyen/MGPATH).

The spatial embedding features can be also download from [Hugging Face](https://huggingface.co/datasets/tiennguyen/MGPATH).


## Model Overview

<p align="center">
    <img src="docs/images/MGPATH-detail.png" width="90%"> <br>

  *Overview of MGPath model architecture*

</p>

## Note

`docs/cmd` provides essential information to play with the source codes.

## Acknowledgement

This work is supported in part by funds from the German Ministry of Education and Research (BMBF) under grant agreements No. 01D2208A and No. 01KD2414A (project FAIrPaCT). The authors gratefully acknowledge the computing time granted by the KISSKI project. The calculations for this research were conducted with computing resources under the project kisski-umg-fairpact-2. The authors also acknowledge the computing time granted by the Resource Allocation Board and provided on the supercomputer Emmy/Grete at NHR-Nord@Göttingen as part of the NHR infrastructure. The calculations for this research were conducted with computing resources under the project nim00014. 


The project “Development of an intelligent collaboration service for AI-based collaboration between rescue services and central emergency rooms” (acronym: CONNECT_ED) is funded by the German Federal Ministry of Education and Research under grant number 16SV8977 and by the joint project KISSKI under grant number 1IS22093E. 


The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Duy M. H. Nguyen. Duy M. H. Nguyen and Daniel Sonntag are also supported by the XAINES project (BMBF, 01IW20005), No-IDLE project (BMBF, 01IW23002), and the Endowed Chair of Applied Artificial Intelligence, Oldenburg University.




## Usage and License Notices

The model is not intended for clinical use as a medical device, diagnostic tool, or any technology for disease diagnosis, treatment, or prevention. It is not a substitute for professional medical advice, diagnosis, or treatment. Users are responsible for evaluating and validating the model to ensure it meets their needs before any clinical application.

## Citation
If MGPath is useful for yoru research and applications, please cite using this Bibtex:

```bibtex
@article{
    nguyen2025mgpath,
    title={{MGPATH}: A Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot Whole Slide Pathology Classification},
    author={Anh-Tien Nguyen and Duy Minh Ho Nguyen and Nghiem Tuong Diep and Trung Quoc Nguyen and Nhat Ho and Jacqueline Michelle Metsch and Miriam Cindy Maurer and Daniel Sonntag and Hanibal Bohnenberger and Anne-Christin Hauschild},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=u7U81JLGjH},
    note={}
}
```
