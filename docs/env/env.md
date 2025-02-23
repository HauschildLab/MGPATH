# Development Environment Installation

## Prerequisite

python==3.10.13

cuda==11.8.0

## Python virtual environment

Inside the directory MGPATH

```bash
python3 -m venv venv
```

Activate virtual environment

```bash
source venv/bin/activate
```

## Install Pytorch v2.3.1

```bash
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
