import torch
import requests

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria


def get_bunny_model_type(model_name):
    if "phi-1.5" in model_name:
        return "phi-1.5"
    elif "phi-2" in model_name:
        return "phi-2"
    elif "stablelm-2" in model_name:
        return "stablelm-2"
    elif "bunny-v1_0-3b" in model_name.lower():
        return "phi-2"
    else:
        raise ValueError(f"Model type not found for model: {model_name}")


class Bunny:
    def __init__(self, model_path, model_base, model_type):
        """
        Initializes the Bunny model with the specified model path and optional base/type.

        :param model_path: Local path to the model.
        :param model_base: Base model for loading pre-trained weights.
        :param model_type: Type of the model, e.g., 'phi-1.5', 'phi-2', 'stablelm-2'.
        """
        disable_torch_init()

        self.model_path = model_path
        self.model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path, model_base, self.model_name, model_type)

        self.model.eval()


    def generate(self, query_text, query_images, temperature=0.2, max_new_tokens=512):
        conv_mode = "bunny"

        conv = conv_templates[conv_mode].copy()
        roles = conv.roles

        inp = query_text
            
        if query_images is not None:
            image_tensor = process_images([query_images], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=self.model.dtype) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=self.model.dtype)
 
            # first message
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
        if query_images is not None:        
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if temperature > 1e-5 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
        else:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True if temperature > 1e-5 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

        answer_text = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()


        return prompt, answer_text