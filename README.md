# SpatialEval

Welcome to the official codebase for [Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models](https://arxiv.org/abs/2406.14852). 

[[📖 Paper](https://arxiv.org/pdf/2406.14852)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/MilaWang/SpatialEval)]


## 🚀 Quick Start



### 📍 Load Dataset

SpatialEval provides three input modalities—TQA (Text-only), VQA (Vision-only), and VTQA (Vision-text)—across four tasks: Spatial-Map, Maze-Nav, Spatial-Grid, and Spatial-Real. Each modality and task is easily accessible via Hugging Face. Ensure you have installed the [packages](https://huggingface.co/docs/datasets/en/quickstart):

```python
from datasets import load_dataset

tqa = load_dataset("MilaWang/SpatialEval", "tqa", split="test")
vqa = load_dataset("MilaWang/SpatialEval", "vqa", split="test")
vtqa = load_dataset("MilaWang/SpatialEval", "vtqa", split="test")
```


### 📈 Evaluate SpatialEval

SpatialEval supports any evaluation pipelines compatible with language models and vision-language models. For text-based prompts, use the `text` column with this structure:
`{text} First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation.` The image input is in the `image` column, and the correct answers are available in the `oracle_answer`, `oracle_option`, and `oracle_full_answer` columns.

Next, we provide full scripts for inference and evaluation.

#### Install

1. Clone this repository

```python
git clone git@github.com:jiayuww/SpatialEval.git
```

2. Install dependencies

To run models like LLaVA and Bunny, install [LLaVA](https://github.com/haotian-liu/LLaVA) and [Bunny](https://github.com/BAAI-DCAI/Bunny). Install [fastchat](https://github.com/lm-sys/FastChat) for language model inference.
For Bunny variants, ensure you merge LoRA weights into the base LLMs before initiation.

#### 💬 Inference

For example, to run on Llama-3-8B for all four tasks:

```bash
# inference TQA on all tasks
python inference_lm.py --task "all" --mode "tqa" --w_reason --model-path "meta-llama/Meta-Llama-3-8B-Instruct" --output_folder outputs --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
# inference TQA on Spatial-Map
python inference_lm.py --task "spatialmap" --mode "tqa" --w_reason --model-path "meta-llama/Meta-Llama-3-8B-Instruct" --output_folder outputs --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
# inference TQA on Maze-Nav
python inference_lm.py --task "mazenav" --mode "tqa" --w_reason --model-path "meta-llama/Meta-Llama-3-8B-Instruct" --output_folder outputs --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
# inference TQA on Spatial-Grid
python inference_lm.py --task "spatialgrid" --mode "tqa" --w_reason --model-path "meta-llama/Meta-Llama-3-8B-Instruct" --output_folder outputs --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
# inference TQA on Spatial-Real
python inference_lm.py --task "spatialreal" --mode "tqa" --w_reason --model-path "meta-llama/Meta-Llama-3-8B-Instruct" --output_folder outputs --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
```

To run LLaVA-1.6-Mistral-7B across all tasks:

```python
# inference VQA on all tasks
python inference_vlm.py --mode "vqa" --task "all" --model_path "liuhaotian/llava-v1.6-mistral-7b" --w_reason --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
# inference VTQA on all tasks
python inference_vlm.py --mode "vtqa" --task "all" --model_path "liuhaotian/llava-v1.6-mistral-7b" --w_reason --temperature 0.2 --top_p 0.9 --repetition_penalty 1.0 --max_new_tokens 512 --device "cuda"
```

Example bash scripts are available in the `scripts/` folder. For more configurations, see `configs/inference_configs.py`. VLMs support `tqa`, `vqa`, and `vtqa` modes, while LMs support `tqa` only. Tasks include all four tasks or individual tasks like `spatialmap`, `mazenav`, `spatialgrid`, and `spatialreal`.
We can also test the first `k` examples, for exmaple, first 100 samples for each question type in each task by specifying `--first_k 100`.

#### 📊 Evaluation

We use exact match for evaluation. For example, to evaluate Spatial-Map task on all three input modalities TQA, VQA and VTQA:

```bash
# For TQA on Spatial-Map
python evals/evaluation.py --mode 'tqa' --task 'spatialmap' --output_folder 'outputs/' --dataset_id 'MilaWang/SpatialEval' --eval_summary_dir 'eval_summary'
# For VQA on Spatial-Map
python evals/evaluation.py --mode 'vqa' --task 'spatialmap' --output_folder 'outputs/' --dataset_id 'MilaWang/SpatialEval' --eval_summary_dir 'eval_summary'
# For VTQA on Spatial-Map
python evals/evaluation.py --mode 'vtqa' --task 'spatialmap' --output_folder 'outputs/' --dataset_id 'MilaWang/SpatialEval' --eval_summary_dir 'eval_summary'
```

Evaluation can also be configured for other tasks `mazenav`, `spatialgrid`, and `spatialreal`. Further details are in `evals/evaluation.py`.

### 💡 Dataset Generation Script

Stay tuned! The dataset generation script will be released in Feburary 😉

## ⭐ Citation

If you find our work helpful, please consider citing our paper 😊

```
@article{wang2024spatial,
  title={Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models},
  author={Wang, Jiayu and Ming, Yifei and Shi, Zhenmei and Vineet, Vibhav and Wang, Xin and Li, Yixuan and Joshi, Neel},
  journal={arXiv preprint arXiv:2406.14852},
  year={2024}
}
```