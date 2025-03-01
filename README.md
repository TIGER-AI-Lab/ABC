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

- We introduce ABC, an open-source multimodal embedding model that uses a
vision-language model backbone to deeply integrate image features with natural language
instructions.
- ABC achieves best-for-size performance on MSCOCO image-to-text retrieval and is the
top performing model on zero-shot classification and VQA tasks in the Massive Multimodal Embedding
Benchmark.
- Due to its unique novel instruction finetuning regime, ABC excels at using instructions to solve subtle and potentially ambiguous visual retrieval problems.
- To evaluate this capability, we design `CtrlBench`, a benchmark that requires
interleaving textual instructions with image content for correct retrieval.
ABC advances the state of multimodal embeddings by offering both high-quality
representations and flexible natural language control.

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