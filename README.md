# ABC: Achieving Better Control of Multimodal Embeddings using VLMs
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>

<br>

<span style="font-size: 14pt; font-family: Roboto, Helvetica, Arial, Heveltica Neue, sans-serif">
     <b>Authors:</b>
     <a class="name" target="_blank" href="https://benjaminschneider.ca/">Benjamin Schneider</a>, 
     <a class="name" target="_blank" href="https://cs.uwaterloo.ca/~fkerschb/">Florian Kerschbaum</a>,
     <a class="name" target="_blank" href="https://wenhuchen.github.io/">Wenhu Chen</a>&nbsp; @ 
     <a class="btna" target="_blank" href="https://huggingface.co/TIGER-Lab">TIGER-Lab</a> &nbsp; 
     </span>

## 🔥News

- [2025/2/26] Release of the [ABC Paper](LINK_HERE), along with the first release of our [🤗 Model and Datasets](LINK_HERE - make a collection) on Hugging Face (more to come, stay tuned!).


## Overview
![./assets/images/ac_overview.png](./assets/images/ac_overview.png)

<details><summary>Abstract</summary> 

- We introduce AceCoder, the first work to propose a fully automated pipeline for synthesizing large-scale reliable tests used for the reward model training and reinforcement learning in the coding scenario. To do this, we curated the dataset [AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K), where we start from a seed code dataset and prompt powerful LLMs to "imagine" proper test cases for the coding question and filter the noisy ones.

- We trained two reward model [AceCodeRM-7B](https://huggingface.co/TIGER-Lab/AceCodeRM-7B) and [AceCodeRM-32B](https://huggingface.co/TIGER-Lab/AceCodeRM-32B) on the constructed [preference pairs](https://huggingface.co/datasets/TIGER-Lab/AceCodePair-300K). Best-of-N sampling results on HumanEval(+), MBPP(+), BigCodeBench, LiveCodeBench (V4) show consistent improvement.

- We perform RL training from three policy models: Qwen2.5-7B-Instruct and Qwen2.5-Coder-7B-Base and Qwen2.5-Coder-7B-Instruct. Two types of reward can be used, i.e. the trained reward model RM-7B and the rule-based reward, i.e. binary pass rate over the test cases in dataset. Additionaly, we also experiment with RL from the base model like DeepSeek-R1. Results show that directly RL from the Base Qwen2.5-Coder model can get **25%** improvement on HumanEval-plus and **6%** on MBPP-plus within just **80** optimization steps.

- To our knowledge, this is the first work to propose a fully automated pipeline for synthesizing large-scale reliable tests used for the reward model training and reinforcement learning in the coding scenario. We believe our \dataset{} will unlock the potential of RL training for code generation models and help the community to further push the boundaries of LLM's coding abilities.

</details>

## 🤗 Models

| Model | Supports Instructions | Base Model | Training Dataset |
|:---------------------:|:-----------:|:----------------:|:--------------:|
| ABC-Qwen2VL-Instruct  | [x]         | ABC-Qwen2VL-Pretrain | [TIGER-Lab/ABC-VG-Instruct]() |
| ABC-Qwen2VL-Pretrain  | [ ]         | Qwen2VL-Instruct     | [TIGER-Lab/ABC-Pretrain]()    |

## 📚 Datasets
- [ABC-VG-Instruct](): A custom dataset for multimodal finetuning. Contains multiple instructions per image, each corresponding to different aspects of each image.
- [ABC-Pretrain](): Multimodal pretraining dataset with mined negatives.


## 🚀 Quick Start

Install Dependancies:
```bash
git clone $
cd ABC
pip install -r requirements.txt
```
Start making multimodal embeddings!
```bash
python -i ./quick_start.py
```

## 📈 Zero-shot Performance
Check out our [paper]() for additional evaluations!

### Instruction Finetuning
Our training requires a slightly larger set of dependancies, please install them:
```bash
git pip install -r training_requirements.txt
```

### Pretraining
Coming soon! Our pretraining dataset is relatively large (~300 GB), so we are currently exploring how to offer downloading as easy as possible. Currently you can download the images in the pretraining from the `url` field in each row.

## Citation
If you find this work helpful, please consider citing:
```bibtex
@article{}
```