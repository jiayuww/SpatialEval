import argparse
import torch
import json
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from fastchat.model import load_model, get_conversation_template, add_model_args
from utils.format_filename import format_output_path_lm
from datasets import load_dataset
from configs.inference_configs import InferenceArgumentParser

def load_model_tokenizer_adapted(args):
    model, tokenizer = load_model(
            args.model_path,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            revision=args.revision,
            dtype=args.dtype
        )
    return model, tokenizer


def load_model_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return model, tokenizer

@torch.inference_mode()
def main(args, model, tokenizer, dataset, output_file_path):
    question_groups = {}

    for item in dataset:
        question_id = item['id'].split('.')[-1]

        if question_id not in question_groups:
            question_groups[question_id] = []
        
        question_groups[question_id].append(item)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for question_id, items in question_groups.items():
            num_to_process = args.first_k if args.first_k is not None else len(items)
            for index, item in enumerate(items[:num_to_process]):
                id = item['id']
                msg = item['text']

                if args.completion:
                    msg = f"{msg} Answer:"
                elif args.w_reason:
                    msg = f"{msg} First, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."

                conv = get_conversation_template(args.model_path)
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                inputs = tokenizer([prompt], return_tensors="pt").to(args.device)

                output_ids = model.generate(
                    **inputs,
                    do_sample=True if args.temperature > 1e-5 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )

                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(inputs["input_ids"][0]):]
                outputs = tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )

                result = {
                    "id": id, 
                    "answer": outputs, 
                    "oracle_answer": item['oracle_answer'], 
                    "oracle_option": item['oracle_option'], 
                    "oracle_full_answer": item['oracle_full_answer'], 
                    "prompt": msg
                }

                json_record = json.dumps(result)
                outfile.write(json_record + '\n')
                outfile.flush()
                os.fsync(outfile.fileno())

                if index % 10 == 0:
                    print(f"Processed {index} items.")
                    print(f"{conv.roles[0]}: {msg}")
                    print(f"{conv.roles[1]}: {outputs}")

    print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    args = InferenceArgumentParser("lm").parse_args()
    dataset = load_dataset(args.dataset_id, args.mode, split="test")
    
    if args.task != "all":
        dataset = dataset.filter(lambda x: args.task in x['id'])
    else:
        dataset = dataset

    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    elif "mistral" in args.model_path.lower():
        model, tokenizer = load_model_tokenizer(args)
    else:
        model, tokenizer = load_model_tokenizer_adapted(args)

    output_path = format_output_path_lm(args)

    main(args, model, tokenizer, dataset, output_path)