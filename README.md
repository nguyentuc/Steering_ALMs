# Activation Steering for LALMs Reasoning 


## Environment Setup
```
conda create -p /media/volume/h100_instance2/conda_env/ALMs_Steering python=3.10.14 -y
conda activate /media/volume/h100_instance2/conda_env/ALMs_Steering
pip install -r requirements.txt
pip uninstall torch torchaudio torchvision -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets soundfile
pip install torchcodec
pip install gdown
pip install 'accelerate>=0.26.0'
pip install matplotlib tiktoken
conda install -c conda-forge ffmpeg
```

## Notes
- Prompt: https://github.com/xzf-thu/Audio-Reasoner
- Models load: https://github.com/QwenLM/Qwen-Audio; https://github.com/QwenLM/Qwen2-Audio

## Next step:
1. Evaluate current LALMs on the MMAU test set.
2. Apply steering on the LALMs.
3. Second reasoning benchmark ReasonAQA: https://github.com/soham97/mellow?tab=readme-ov-file#reasonaqa
