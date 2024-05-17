# APD (WIP)
Extending ["APE: Aligning Pretrained Encoders to Quickly Learn Aligned Multimodal Representations"](https://arxiv.org/abs/2210.03927) to use decoders via ["LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"](https://arxiv.org/abs/2404.05961).


### Setup
```bash
conda create -n "apd" python=3.10
conda activate apd

conda install nvidia::cuda-toolkit
export CUDA_HOME=CONDA_PREFIX

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers
pip install open_clip_torch

cd llm2vec
pip install -e .

pip install flash-attn --no-build-isolation
```

### Current Results
- Results collected using [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- Finetuned 4-layer MLP for 1 epoch on coco 2017 train set (591753 samples with batch size of 256)

**Zero-shot Classification**
| Text Encoder | Vision Encoder + Pretrained | Dataset | acc1 | acc5 | mean_per_class_recall |
|-------------|-------------|:-------------:|:-------------:|:-------------:|:-------------:|
| LLM2Vec-Sheared-LLaMA-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | cifar10 | 0.9087 | 0.9948 | 0.9083 |
| LLM2Vec-Sheared-LLaMA-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | cifar100 | 0.4747 | 0.7623 | 0.475 |
| LLM2Vec-Sheared-LLaMA-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | imagenet1k | 0.2544 | 0.5058 | 0.2544 |
| LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | cifar10 | 0.8942 | 0.9962 | 0.8942 |
| LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | cifar100 | 0.4938 | 0.7697 | 0.4938 |
| LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | imagenet1k | 0.2403 | 0.4991 | 0.2406 |

**Zero-shot Retrieval**
| Text Encoder | Vision Encoder + Pretrained | Dataset | image_retrieval_recall@5 | text_retrieval_recall@5 |
|-------------|-------------|:-------------:|:-------------:|:-------------:|
| LLM2Vec-Sheared-LLaMA-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | mscoco_captions | 0.6604 | 0.7982 |
| LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised | ViT-B-32 + laion2b_s34b_b79k | mscoco_captions | 0.6519 | 0.7874 |


### Citations
```bibtex
@article{llm2vec,
    title={{LLM2Vec}: {L}arge Language Models Are Secretly Powerful Text Encoders}, 
    author={Parishad BehnamGhader and Vaibhav Adlakha and Marius Mosbach and Dzmitry Bahdanau and Nicolas Chapados and Siva Reddy},
    year={2024},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/2404.05961}
}

@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  year={2023}
}

@software{Ilharco_OpenCLIP_2021,
    author = {Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
    doi = {10.5281/zenodo.5143773},
    month = jul,
    title = {{OpenCLIP}},
    version = {v0.1},
    year = {2021}
}
```
