# StepNav: Efficient Planning with Structured Trajectory Priors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ICRA 2026](https://img.shields.io/badge/ICRA-2026-blue.svg)](https://2026.ieee-icra.org/)

**[Project Page](https://luoxubo.github.io/StepNav/)** | **[Paper](https://arxiv.org/abs/XXXX.XXXXX)** | **[Code](https://github.com/LuoXubo/StepNav)**

> Official PyTorch implementation of **StepNav**, an efficient visual navigation framework accepted at **ICRA 2026**.

## Abstract

We present StepNav, an efficient planning framework for visual navigation that generates reliable trajectories using structured trajectory priors. Unlike existing methods that rely on unstructured noise, StepNav leverages multi-modal trajectory initialization combined with conditional flow matching for efficient and safe path generation.

![StepNav Overview](docs/static/images/pipeline.png)

## Installation

```bash
git clone https://github.com/LuoXubo/StepNav.git
cd StepNav
conda create -n stepnav python=3.8
conda activate stepnav
pip install -r requirements.txt
pip install -e .
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{luo2026stepnav,
  title={StepNav: Efficient Planning with Structured Trajectory Priors},
  author={Luo, Xubo and Wu, Aodi and Han, Haodong and Wan, Xue and Zhang, Wei and Shu, Leizheng and Wang, Ruisuo},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

## Acknowledgments

This work builds upon [NoMaD](https://github.com/robodhruv/visualnav-transformer), [Flownav](https://github.com/utn-air/flownav), and [NaviBridger](https://github.com/hren20/NaiviBridger).