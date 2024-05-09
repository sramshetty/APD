# APD

### Setup
```bash
conda create -n "apd" python=3.10
conda activate apd

conda install nvidia::cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install transformers

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
```