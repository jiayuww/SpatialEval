import os
import argparse
from pathlib import Path

def format_output_path_vlm(args):
    file_ext = ".jsonl"
    if hasattr(args, 'model_id') and args.model_id is not None:
        model_name = args.model_id.replace("/", "__")
    elif hasattr(args, 'model_path') and args.model_path is not None:
        model_name = args.model_path.replace("/", "__")
    else:
        raise ValueError("Both model_id and model_path are missing or None.")

    output_dir = Path(args.output_folder) / args.dataset_id.replace("/", "__") / args.mode / args.task

    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename= f"m-{model_name}"

    if args.random_image:
        filename += "_random_image"
    elif args.noise_image:
        filename += "_noise_image"
    if args.w_reason:
        filename += "_w_reason"
    elif args.completion:
        filename += "_completion"
    else:
        filename += "_bare"

    modified_output_filename = f"{filename}{file_ext}"

    output_path = output_dir / modified_output_filename

    # print(f"output_path: {output_path}")

    return output_path

def format_output_path_lm(args):
    file_ext = ".jsonl"
    model_name = args.model_path.replace("/", "__")
    
    output_suffix = ""

    output_dir = Path(args.output_folder) / args.dataset_id.replace("/", "__") / args.mode / args.task

    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.completion:
        output_suffix += "_completion"
    elif args.w_reason:
        output_suffix += "_w_reason"
    else:
        output_suffix += "_bare"

    modified_output_filename = f"m-{model_name}{output_suffix}{file_ext}"
    
    output_path = output_dir / modified_output_filename

    # print(f"output_path: {output_path}")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="MilaWang/SpatialEval",
                                 help="Dataset identifier for Hugging Face.")
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_folder", type=str, default="/data/milawang/SpatialEval/outputs")
    parser.add_argument("--task", type=str, default="all", choices=["all", "spatialmap", "mazenav", "spatialgrid", "spatialreal"],
                                help="Set specific task to evaluate or evaluate all tasks.")
    parser.add_argument("--completion", action="store_true", help="Add completion prompt.")
    parser.add_argument("--w_reason", action="store_false", help="Add reason prompt.")
    parser.add_argument("--mode", default="vqa", choices=["tqa", "vqa", "vtqa"], 
                                 help="Set mode for test input modality (only 'tqa' allowed for language model).")
    parser.add_argument("--random_image", action="store_false", help="Use random image for inference.")
    parser.add_argument("--noise_image", action="store_true", help="Use noise image for inference.")
    args = parser.parse_args()

    # format_output_path_lm(args)
    format_output_path_vlm(args)