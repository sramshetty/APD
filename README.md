# APD

### Setup
```bash
conda create -n "apd" python=3.10
conda activate apd

conda install nvidia::cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers
pip install open_clip_torch

cd llm2vec
pip install -e .

cd ../flash-attention
python setup.py install
```


### Citations
```bibtex
@article{llm2vec,
    title={{LLM2Vec}: {L}arge Language Models Are Secretly Powerful Text Encoders}, 
    author={Parishad BehnamGhader and Vaibhav Adlakha and Marius Mosbach and Dzmitry Bahdanau and Nicolas Chapados and Siva Reddy},
    year={2024},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/2404.05961}
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