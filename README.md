This is the code repo for *Advancing Abductive Reasoning in Knowledge Graphs through Complex Logical Hypothesis Generation*
- Paper: https://arxiv.org/abs/2312.15643
- Slides: [Google drive](https://drive.google.com/file/d/1IqxEQFN-ofk8-ODrESjitX7Lmi0oVb7P/view?usp=sharing)

# Environment

```bash
conda create -n akgr python=3.10
conda activate akgr
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
```

# Training

As described in the paper, the Reinforcement Learning from Knowledge
Graph Feedback (RLF-KG) pipeline comprises the following steps:
1. Sampling
2. Supervised training
3. Reinforcement learning

## Step 1: Sampling

```bash
bash scripts/sample/sample_full.sh
```

See [Example Data and Checkpoints](#example-data-and-checkpoints)

## Step 2: Supervised training

Example scripts:

```bash
bash scripts/train/fb-t5.sh
bash scripts/train/db-t5.sh
bash scripts/train/wn-t5.sh
bash scripts/train/fb-g2.sh
bash scripts/train/db-g2.sh
bash scripts/train/wn-g2.sh
```

## Step 3: Reinforcement learning

Example scripts:

```bash
bash scripts/optim/fb-t5-0.0.sh
bash scripts/optim/fb-t5-0.2.sh
bash scripts/optim/db-t5-0.0.sh
bash scripts/optim/db-t5-0.2.sh
bash scripts/optim/wn-t5-0.0.sh
bash scripts/optim/wn-t5-0.2.sh
bash scripts/optim/fb-g2-0.0.sh
bash scripts/optim/fb-g2-0.2.sh
bash scripts/optim/db-g2-0.0.sh
bash scripts/optim/db-g2-0.2.sh
bash scripts/optim/wn-g2-0.0.sh
bash scripts/optim/wn-g2-0.2.sh
```

# Evaluation

Example scripts:

```bash
bash scripts/test/fb-t5.sh
bash scripts/test/db-t5.sh
bash scripts/test/wn-t5.sh
bash scripts/test/fb-g2.sh
bash scripts/test/db-g2.sh
bash scripts/test/wn-g2.sh
```

```bash
bash scripts/optim-test/fb-t5-0.0.sh
bash scripts/optim-test/fb-t5-0.2.sh
bash scripts/optim-test/db-t5-0.0.sh
bash scripts/optim-test/db-t5-0.2.sh
bash scripts/optim-test/wn-t5-0.0.sh
bash scripts/optim-test/wn-t5-0.2.sh
bash scripts/optim-test/fb-g2-0.0.sh
bash scripts/optim-test/fb-g2-0.2.sh
bash scripts/optim-test/db-g2-0.0.sh
bash scripts/optim-test/db-g2-0.2.sh
bash scripts/optim-test/wn-g2-0.0.sh
bash scripts/optim-test/wn-g2-0.2.sh
```

See [Example Data and Checkpoints](#example-data-and-checkpoints)

# Example Data and Checkpoints

Sampled data: Download [Onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ywangmy_connect_ust_hk/EtvmEUWl-dxBgfYckjLqIsUBLjjs2_WvZB2IWNLDmwAnyw?e=2t4RoH) to `sampled_data` under the root.

Checkpoints: Download [Onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ywangmy_connect_ust_hk/EpSiNlWJ_uROuYZZS2mhy1wB2l5A8RKgyAnuL-hnbenyRQ?e=LdXbyA) to `checkpoints` under the root.


# Citation

```bib
@misc{bai2024advancingabductivereasoningknowledge,
      title={Advancing Abductive Reasoning in Knowledge Graphs through Complex Logical Hypothesis Generation}, 
      author={Jiaxin Bai and Yicheng Wang and Tianshi Zheng and Yue Guo and Xin Liu and Yangqiu Song},
      year={2024},
      eprint={2312.15643},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2312.15643}, 
}
```