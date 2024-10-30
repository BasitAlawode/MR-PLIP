import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

#questions = ["Can you describe the main features visible in this histopathology image?",
#             "Can you provide a concise description of the histopathology image shown?"]
#image_folder = "test_images"
#answers_dir = [f"{image_folder}_answers/main",
#               f"{image_folder}_answers/concise"]

questions = ["Can you describe the main features visible in this histopathology image?"]

image_folder = "test_images_sup"
answers_dir = [f"{image_folder}_answers/main"]

ckpt = "wisdomik/Quilt-Llava-v1.5-7b"
temp, conv_mode = 0, "vicuna_v1"

for a in answers_dir:
    if not os.path.exists(a): os.makedirs(a) 

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    args.model_path = ckpt
    args.temperature = temp
    args.conv_mode = conv_mode
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    
    for qs, ans_dir in zip(questions, answers_dir):
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        for image_path in [f"{image_folder}/{a}" for a in os.listdir(image_folder)]:
            image = Image.open(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            
            image_name = image_path.split('/')[-1].split('.')[0]
            ans_file = open(f"{ans_dir}/{image_name}.jsonl", "w")
            ans_file.write(json.dumps({"Image": f"{image_name}.PNG",
                                       "prompt": cur_prompt,
                                        "Answer": outputs,
                                        "model_id": model_name}))
            #ans_file.flush()
            ans_file.close()
    
    print('Done')
