# Adaptation with Noisy OracLE (ANOLE)

This repository contains a PyTorch implementation of our paper [Efficient Meta Reinforcement Learning for Preference-based Fast Adaptation](https://arxiv.org/abs/2211.10861) published at NeurIPS 2022.

## Installation

Clone this repository and set up the dependencies using the following commands:

```bash
git clone --recurse-submodules https://github.com/Stilwell-Git/Adaptation-with-Noisy-OracLE.git
cd Adaptation-with-Noisy-OracLE
conda create -n anole python=3.8
conda activate anole
conda install pytorch-gpu=1.10 -c conda-forge
conda install click joblib python-dateutil
pip install gym[mujoco]==0.12.1
pip install gtimer
```

## Running Commands

Run the following commands to reproduce our main results shown in section 4.2.

```bash
python launch_anole_experiment.py --config anole/configs/halfcheetah-fwd-back.json  # HalfCheetah-Fwd-Back
python launch_anole_experiment.py --config anole/configs/halfcheetah-rand-vel.json  # HalfCheetah-Rand-Vel
python launch_anole_experiment.py --config anole/configs/walker2d-rand-vel.json     # Walker2d-Rand-Vel
python launch_anole_experiment.py --config anole/configs/ant-fwd-back.json          # Ant-Fwd-Back
python launch_anole_experiment.py --config anole/configs/ant-rand-dir.json          # Ant-Rand-Dir
python launch_anole_experiment.py --config anole/configs/ant-rand-goal.json         # Ant-Rand-Goal
```

Please use argument `--gpu $GPU_ID` to specify the GPU device.