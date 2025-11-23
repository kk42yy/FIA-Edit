# FIA-Edit

<p align="center">
  <a href="https://arxiv.org/abs/2511.12151">
    <img src="https://img.shields.io/badge/arXiv-2511.12151-b31b1b?logo=arxiv">
  </a>
</p>


[AAAI 2026] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing

## 🚀 Getting Started
<span id="getting-started"></span>

### Environment
<span id="environment-requirement"></span>
```shell
conda create -n FSI-Edit python=3.12 -y
conda activate FSI-Edit
conda install conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Benchmark Download ⬇️
<span id="benchmark-download"></span>
Please refer to [PIE-Bench](https://github.com/cure-lab/PnPInversion#benchmark-download).


## Running Scripts
<span id="running-scripts"></span>

### Inference
<span id="inference"></span>

**Run PIE-Bench**
```shell
source Infer.sh
```

**Run Single Image**
```shell
python run_FIA-Edit.py
```

### Evaluation
<span id="evaluation"></span>

```shell
source Evaluate.sh
```

### Comparison Results
FlowEdit and Our results are [here](https://huggingface.co/datasets/kk42yy/FIA-Edit), the rest of our comparision results can be found [FSI-Edit](https://huggingface.co/datasets/kk42yy/FSI-Edit) and [PnP Inversion](https://github.com/cure-lab/DirectInversion).

## 🤝🏼 Cite Us
```
@article{yang2025fia,
  title={FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing},
  author={Yang, Kaixiang and Shen, Boyang and Li, Xin and Dai, Yuchen and Luo, Yuxuan and Ma, Yueran and Fang, Wei and Li, Qiang and Wang, Zhiwei},
  journal={arXiv preprint arXiv:2511.12151},
  year={2025}
}
```


## 💖 Acknowledgement
<span id="acknowledgement"></span>
- [PnP Inversion](https://github.com/cure-lab/DirectInversion)
- [FlowEdit](https://github.com/fallenshock/FlowEdit)
- [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play)
- [FlexiEdit](https://github.com/kookie12/FlexiEdit)
- [FreeDiff](https://github.com/thermal-dynamics/freediff)
- [FSI-Edit](https://github.com/kk42yy/FSI-Edit)
