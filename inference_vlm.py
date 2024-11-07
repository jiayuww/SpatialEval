from PIL import Image
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, InstructBlipProcessor, InstructBlipForConditionalGeneration
import json
import os
import warnings
from utils.format_filename import format_output_path_vlm
from configs.inference_configs import InferenceArgumentParser
from datasets import load_dataset

IMAGE_TOKEN_INDEX = -200

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def format_bunny_vqa_prompt_hf(text, args):
    if args.w_reason:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text}\nFirst, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation. ASSISTANT:"
    elif args.completion:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text} Answer: ASSISTANT:"
    else:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{text} ASSISTANT:"

def format_bunny_tqa_prompt_hf(text, args):
    if args.w_reason:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text}\nFirst, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation. ASSISTANT:"
    elif args.completion:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text}\n Answer: ASSISTANT:"
    else:
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text} ASSISTANT:"

def load_bunny_model_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    return model, tokenizer

def load_qwen_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, trust_remote_code=True).eval()

    return model, tokenizer

def load_cog_model_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device).eval()
    tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    return model, tokenizer

def load_instructblip_model_processor(args):
    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    processor = InstructBlipProcessor.from_pretrained(args.model_path)

    return model, processor

@torch.inference_mode()
def main(args, model, processor, dataset, output_file_path):
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
                
                if args.mode == "tqa":
                    image_path = None
                else:
                    if args.random_image:
                        if "mazenav" in id.lower():
                            image_path = "assets/random_maze_nav.png"
                        elif "spatialgrid" in id.lower():
                            image_path = f"assets/random_spatial_grid.png"
                        elif "spatialmap" in id.lower():
                            image_path = "assets/random_spatial_map.png"
                        else:
                            raise ValueError(f"Unknown dataset type for random image: {args.task}")
                    elif args.noise_image:
                        image_path = "assets/noise.png"
                    else:
                        image_path = item['image']
                if args.mode == "tqa":
                    image = None
                else:
                    image = load_image(image_path)
                if "bunny" in args.model_path.lower() and "merged" not in args.model_path.lower():
                    if args.mode == "tqa":
                        prompt = format_bunny_tqa_prompt_hf(item['text'], args)
                    else:
                        prompt = format_bunny_vqa_prompt_hf(item['text'], args)
                elif "qwen" or "cog" or "instructblip" or "llava" in args.model_path.lower() or "merged" in args.model_path.lower():
                    if args.w_reason:
                        prompt = f"{item['text']}\nFirst, provide a concise answer in one sentence. Then, elaborate on the reasoning behind your answer in a detailed, step-by-step explanation."
                    elif args.completion:
                        prompt = f"{item['text']}\nAnswer:"
                    else:
                        prompt = item['text']
                else:
                    raise ValueError(f"The maze dataset does not support the model {args.model_path}.")

                if "bunny" in args.model_path.lower() and "merged" not in args.model_path.lower():
                    if image is not None:
                        text_chunks = [processor(chunk).input_ids for chunk in prompt.split('<image>')]
                        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)

                        image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True)[0]

                        answer_text = processor.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
                    else:
                        input_ids = tokenizer_image_token(prompt, processor, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                        0).to(model.device)
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True)[0]

                        answer_text = processor.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

                elif "qwen" in args.model_path.lower():
                    if args.mode == "tqa":
                        query = processor.from_list_format([{'text': prompt}])
                    else:
                        query = processor.from_list_format([
                            {'image': image_path},
                            {'text': prompt},
                        ])
                    answer_text, history = model.chat(processor, query=query, history=None)
                elif "cog" in args.model_path.lower():
                    history = []
                    if args.mode == "tqa":
                        text_only_first_query = True
                        
                    history = []
                    if image is None:
                        if text_only_first_query:
                            text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
                            query = text_only_template.format(prompt)
                            text_only_first_query = False
                        else:
                            old_prompt = ''
                            for _, (old_query, response) in enumerate(history):
                                old_prompt += old_query + " " + response + "\n"
                            query = old_prompt + "USER: {} ASSISTANT:".format(query)

                    if image is None:
                        input_by_model = model.build_conversation_input_ids(processor, query=query, history=history, template_version='base')
                    else:
                        input_by_model = model.build_conversation_input_ids(processor, query=prompt, history=history,
                                                                        images=[image])
                    inputs = {
                        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(args.device),
                        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(args.device),
                        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(args.device),
                        'images': [[input_by_model['images'][0].to(args.device).to(torch.bfloat16)]] if image is not None else None,
                    }
                    if 'cross_images' in input_by_model and input_by_model['cross_images']:
                        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(args.device).to(torch.bfloat16)]]

                    # add any transformers params here.
                    gen_kwargs = {"max_length": 2048,
                                "do_sample": False}  # "temperature": 0.9
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **gen_kwargs)
                        outputs = outputs[:, inputs['input_ids'].shape[1]:]
                        answer_text = processor.decode(outputs[0])
                        answer_text = answer_text.split("</s>")[0]

                elif "instructblip" in args.model_path.lower():
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device) # image input is necessary

                    output_id = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_new_tokens=args.max_new_tokens,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                    )
                    answer_text = processor.batch_decode(output_id, skip_special_tokens=True)[0].strip()

                elif "llava" in args.model_path.lower() or ("bunny" and "merged" in args.model_path.lower()):
                    prompt, answer_text = model.generate(prompt, image, args.temperature)
                
                else:
                    raise ValueError(f"Model id {args.model_path} is not supported.")

                result = {
                    "id": id, 
                    "answer": answer_text, 
                    "oracle_answer": item['oracle_answer'], 
                    "oracle_option": item['oracle_option'], 
                    "oracle_full_answer": item['oracle_full_answer'], 
                    "prompt": prompt,
                    "image": image_path if isinstance(image_path, str) else ""}
                
                # print(f"result: {result}")
                
                json_record = json.dumps(result)
                outfile.write(json_record + '\n')
                outfile.flush()
                os.fsync(outfile.fileno())

                if index % 10 == 0:
                    print(f"Processed {index} items.")
                    print(f"{prompt}")
                    print(f"{answer_text}")

    print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    args = InferenceArgumentParser("vlm").parse_args()
    dataset = load_dataset(args.dataset_id, args.mode, split="test")
    
    if args.task != "all":
        dataset = dataset.filter(lambda x: args.task in x['id'])
    else:
        dataset = dataset
        
    if args.mode != "tqa":
        from utils.load_image import load_image
        
    if "llava" in args.model_path.lower():
        from models.llava_model import Llava
        model = Llava(args.model_path, args.model_base)
        processor = None
    
    elif "bunny" in args.model_path.lower() and "merged" in args.model_path.lower():
        # support local model, assume we name the model with merged suffix, e.g., bunny-phi-2-eva-merged
        from models.bunny_model import Bunny, get_bunny_model_type
        model_name = args.model_path.split("/")[-1]
        model_type = get_bunny_model_type(model_name)
        model = Bunny(args.model_path, args.model_base, model_type)
        processor = None
        
    elif "bunny" in args.model_path.lower() and "merged" not in args.model_path.lower():
        # generally support bunny models from huggingface
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        torch.set_default_device(args.device)
        model, processor = load_bunny_model_tokenizer(args)
        
    elif "qwen" in args.model_path.lower():
        model, processor = load_qwen_model_tokenizer(args)
        
    elif "cog" in args.model_path.lower():
        model, processor = load_cog_model_tokenizer(args)
        
    elif "instructblip" in args.model_path.lower():
        model, processor = load_instructblip_model_processor(args)
    else:
        raise ValueError(f"Model {args.model_path} is not supported.")

    output_path = format_output_path_vlm(args)
    
    main(args, model, processor, dataset, output_path)