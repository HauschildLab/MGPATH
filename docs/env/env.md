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

## Install pytorch v2.3.1

```bash
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

## Install numpy

```bash
pip3 install numpy==2.1.2
```

## Install torcheval
```bash
pip3 install torcheval==0.0.7
```

## Install transformers
```bash
pip3 install transformers==4.49.0
```

## Install pandas
```bash
pip3 install pandas==2.2.3
```

## Install openai-clip
```bash
pip3 install openai-clip==1.0.1
```

## Install h5py
```bash
pip3 install h5py==3.13.0
```
