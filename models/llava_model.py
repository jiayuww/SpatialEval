import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import re, sys


class Llava:
    def __init__(self, model_path, model_base):
        disable_torch_init()

        self.model_path = model_path
        self.model_name = get_model_name_from_path(self.model_path)

        tokenizer, model, image_processor, _ = load_pretrained_model(self.model_path, model_base, self.model_name)
        model.eval()

        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def generate(self, query_text, query_images=None, temperature=0.2, num_beams=1, max_new_tokens=512):
        qs = query_text

        if query_images is not None:
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            images_tensor = process_images(
                [query_images],
                self.image_processor,
                self.model.config
            ).to(self.model.device, dtype=torch.float16)

            if not isinstance(query_images, list):
                query_images = [query_images]

            image_sizes = [x.size for x in query_images]
        else:
            pass

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        if query_images is not None:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 1e-5 else False,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True if temperature > 1e-5 else False,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

        answer_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return prompt, answer_text
