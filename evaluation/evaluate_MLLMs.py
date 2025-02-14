import argparse
import pandas as pd
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,
    LlavaForConditionalGeneration, LlavaNextProcessor,GenerationConfig,
    LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration,LlavaOnevisionForConditionalGeneration,MllamaForConditionalGeneration, AutoModel, AutoTokenizer, BlipForQuestionAnswering
)
import torch
import argparse
from PIL import Image
import numpy as np
import random
import re
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import ast
from io import BytesIO
import openai
import base64
from math_llava_functions import *

# MATH PUMA IMPORTS
from qwen2 import Qwen2vlmProcessor, Qwen2vlmForConditionalGeneration

# JANUS IMPORTS
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def remove_example(text):
    return re.sub(r'### Example Output:.*?\nIMPORTANT:', 'IMPORTANT:', text, flags=re.DOTALL)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def evaluate_image_with_gpt(image_path, prompt, model_version):
    API_KEY = "YOUR API KEY HERE"

    client = openai.OpenAI(api_key=API_KEY)
    
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
    if model_version == "o1":
        response = client.chat.completions.create(
            model= model_version,  
            messages=[
                {"role": "system", "content": "Analyze the image and answer the question."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]}
            ],
            max_completion_tokens=1000,
            timeout= 5000
        )
        
    else:
        response = client.chat.completions.create(
            model= model_version,  
            messages=[
                {"role": "system", "content": "Analyze the image and answer the question."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]}
            ],
            max_tokens=1000,
            temperature=0,
            top_p=0
        )

    return response.choices[0].message.content

def generate_text_for_rows(df, processor, model, instruction_tokens, task, model_version, tokenizer=None):
    generated_texts = []

    for index, row in df.iterrows():
        SEED = 42
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.empty_cache()

        if model_version == 'internvl':
            # Custom image loading for InternVL
            pixel_values = load_image(row['path'], max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1000, do_sample=False)
            question = row["prompt"]
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            predicted_answer = response

        elif model_version == "gpt-4o" or model_version == "gpt-4-turbo" or model_version == "o1": 
            image_path = row['path']
            prompt = row["prompt"]
            predicted_answer = evaluate_image_with_gpt(image_path, prompt, model_version)
            
        elif model_version == 'blip':
            image_pil = Image.open(row['path'])
            prompt = row["prompt"]
            inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs)
            predicted_answer = processor.decode(outputs[0], skip_special_tokens=True)

        elif model_version == 'qwen':
            raw_image = Image.open(row['path'])
            if raw_image.size[0] < 28 or raw_image.size[1] < 28:
                print(f"Image too small: {raw_image.size}, resizing to minimum dimensions.")
                raw_image = raw_image.resize((max(28, raw_image.size[0]), max(28, raw_image.size[1])))

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["prompt"]},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                images=raw_image,
                return_tensors="pt",
            ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
            outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("\nassistant\n")[-1]

        elif model_version == 'molmo':
            try:
                image = Image.open(row['path'])
                image = image.convert("RGB")
                
                inputs = processor.process(
                    images=[image],
                    text=row["prompt"]
                )
                inputs = {k: v.unsqueeze(0).to(model.device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    output = model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=1000, stop_strings="<|endoftext|>"),
                        tokenizer=processor.tokenizer
                    )
                
                generated_tokens = output[0, inputs['input_ids'].size(1):]
                predicted_answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            except Exception as e:
                print(f"Error processing row {row}: {e}")  # Logs the error for debugging
                predicted_answer = ""
                
        elif model_version in ['math-llava', 'g-llava']:
            prompt = row["prompt"]
            if task == "two_shapes":
                prompt = prompt.split("\n\n### Example")[0]
            
            args = argparse.Namespace(
                query=prompt,  # Modify this with your actual prompt
                conv_mode=None,  # Keep as None to use the auto-detected mode
                temperature=0,  
                top_p=1,  
                num_beams=1,  
                max_new_tokens=1000,  
                image_file=row["path"],  # Update with your actual image path
                sep=","
            )
            
            # Define model name
            if model_version == 'math-llava':
                model_name = "Zhiqiang007/Math-LLaVA"
            else:
                model_name = "renjiepi/G-LLaVA-7B"
            
            tokenizer = processor["tokenizer"]
            image_processor = processor["image_processor"]
            context_len = processor["context_len"]
            
            predicted_answer = evalmodel(args, model_name, tokenizer, model, image_processor, context_len)
            
        elif model_version == 'math-puma':
            raw_image = Image.open(row['path']).convert('RGB')
            system_prompt = ""

            if raw_image.size[0] < 28 or raw_image.size[1] < 28:
                print(f"Image too small: {raw_image.size}, resizing to minimum dimensions.")
                raw_image = raw_image.resize((max(28, raw_image.size[0]), max(28, raw_image.size[1])))

            
            #print("INDEX: ", index)
            #print("IMG TYPE: ", type(raw_image))
            #print(row)
         
            # Conversation setup
            conv = []
            if system_prompt:
                conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

            user_content = [{"type": "text", "text": row['prompt']}]
            user_content.append({"type": "image"}) 

            conv.append({"role": "user", "content": user_content})
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)

            # Process inputs
            inputs = processor(
                text=prompt,
                images=raw_image,
                return_tensors="pt"
            ).to(model.device).to(torch.float16)

            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs, 
                                        max_new_tokens=100, 
                                        do_sample=False,
                                        num_beams=1)

            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("\nassistant\n")[-1]
            #print(predicted_answer)
            #print("-------------------------------------------------")

        elif model_version == 'janus':
            image_path = row['path']
            #prompt = row["prompt"]
            prompt = remove_example(row["prompt"])
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image = f"data:image/jpeg;base64,{image_data}"

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)

            prepare_inputs = processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(model.device)

            # # run image encoder to get the image embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            tokenizer = processor.tokenizer

            # # run the model to get the response
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
            )
            predicted_answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
        
        elif model_version == 'llava-one':
            raw_image=Image.open(row['path'])
            if raw_image.size[0] < 28 or raw_image.size[1] < 28:
                print(f"Image too small: {raw_image.size}, resizing to minimum dimensions.")
                raw_image = raw_image.resize((max(28, raw_image.size[0]), max(28, raw_image.size[1])))
            conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": row["prompt"]},
                            {"type":"image"}
                        ],
                    },
                ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                    text=prompt,
                    images=raw_image,
                    return_tensors="pt",
                ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
            
            outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0] #.split("assistant\n\n")[-1]
        elif model_version == 'llama-3.2':
            raw_image=Image.open(row['path'])
            if raw_image.size[0] < 28 or raw_image.size[1] < 28:
                print(f"Image too small: {raw_image.size}, resizing to minimum dimensions.")
                raw_image = raw_image.resize((max(28, raw_image.size[0]), max(28, raw_image.size[1])))
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["prompt"]},
                        {"type": "image"}
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                    text=prompt,
                    images=raw_image,
                    return_tensors="pt",
                ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
            
            outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("assistant\n\n")[-1]
            

        else:
            image = Image.open(row['path'])
            prompt = row["prompt"]
            inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=1000)

            if model_version == 'llava-1.5':
                predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1]
            elif model_version == 'llava-1.6':
                predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_answer = clean_instruction_tokens(predicted_answer)

        generated_texts.append(predicted_answer)

    df['generated_text'] = generated_texts
    return df

def main():
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-1.5', 'llava-1.6', 'blip', 'qwen', 'internvl', 'gpt-4o', 'gpt-4-turbo', 'o1','llava-one','llama-3.2','math-llava', 'g-llava', 'molmo', 'janus', 'math-puma'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=['shape_id', 'sides_id', 'two_shapes', 'mathverse_CoT', 'abstract','triangle_cross_ABC_123', 'hept_ABC_123'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")

    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if args.model_version == 'llava-1.6':
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb_config, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        instruction_tokens = "[INST] <image>\n"
        end_tokens = "[/INST]"
        
    elif args.model_version == 'llava-1.5':
        processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        model = LlavaForConditionalGeneration.from_pretrained(
            'llava-hf/llava-1.5-7b-hf', quantization_config=bnb_config
        ).to(0)
        instruction_tokens = "USER: <image>\n"
        end_tokens = "\nASSISTANT:"

    elif args.model_version == 'blip':
        processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        instruction_tokens = ""
        end_tokens = ""
    elif args.model_version == "gpt-4o" or args.model_version == "gpt-4-turbo" or args.model_version == "o1": 
        processor = None
        model = None
        instruction_tokens = ""
        end_tokens = ""
        
    elif args.model_version == 'qwen':
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True
        )
        instruction_tokens = ""
        end_tokens = ""
        
    elif args.model_version == 'internvl':
        os.environ["TRANSFORMERS_CACHE"] = "/users/mgolovan/scratch/huggingface"
        path = "OpenGVLab/InternVL2-8B"
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        instruction_tokens = "<image>\n"
        end_tokens = ""
    elif args.model_version == 'llava-one':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True
        )
        instruction_tokens = ""
        end_tokens = ""

    elif args.model_version in ['math-llava', 'g-llava']:
        if args.model_version == 'math-llava':
            model_path = "Zhiqiang007/Math-LLaVA"
        
        elif args.model_version == 'g-llava':
            model_path = "renjiepi/G-LLaVA-7B"
        
        model_base = None  # Change this if fine-tuning
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        
        # Load model
        if "llava" in model_path.lower():
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        
        # Load the tokenizer separately
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        image_processor = None
        if "llava" in model_path.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            
            if mm_use_im_patch_token:
                tokenizer.add_tokens(["<im_patch>"], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens(["<im_start>", "<im_end>"], special_tokens=True)
            
            model.resize_token_embeddings(len(tokenizer))
        
            vision_tower = model.get_vision_tower()
            if hasattr(vision_tower, "load_model"):
                vision_tower.load_model()
            
            vision_tower.to(device=device, dtype=torch.float16)
            image_processor = vision_tower.image_processor
        
        # Set max context length
        context_len = getattr(model.config, "max_sequence_length", 2048)
        
        processor = {
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "context_len": context_len
        }
        
        instruction_tokens = ""
        end_tokens = ""

        
    elif args.model_version == 'llama-3.2':
        processor = AutoProcessor.from_pretrained("ruslanmv/Llama-3.2-11B-Vision-Instruct")
        model = MllamaForConditionalGeneration.from_pretrained(
            "ruslanmv/Llama-3.2-11B-Vision-Instruct", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True
        )
        instruction_tokens = ""
        end_tokens = ""
    elif args.model_version == 'molmo':
        model = AutoModelForCausalLM.from_pretrained("cyan2k/molmo-7B-D-bnb-4bit", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("cyan2k/molmo-7B-D-bnb-4bit", trust_remote_code=True)
        instruction_tokens = ""
        end_tokens = ""
        
    
    elif args.model_version == 'math-puma':
        # Load model and processor in float16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Qwen2vlmForConditionalGeneration.from_pretrained(
            "Math-PUMA/Math-PUMA_Qwen2VL-7B",
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        model.half()
        
        # Things that are supposed to make inference faster....
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model, mode="max-autotune")

        processor = Qwen2vlmProcessor.from_pretrained("Math-PUMA/Math-PUMA_Qwen2VL-7B")    
        instruction_tokens=""
        end_tokens="" 
    
    elif args.model_version == 'janus':
        model_path = "deepseek-ai/Janus-Pro-7B"
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = model.to(torch.bfloat16).cuda().eval()
        instruction_tokens = ""
        end_tokens = ""
        instruction_tokens=""
        end_tokens=""

    if args.task == 'shape_id':
        df = pd.read_csv("all_shapes.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)
        
        df["prompt"] = instruction_tokens + "What shape is in the image? Answer with one word." + end_tokens 

    elif args.task == 'sides_id':
        df = pd.read_csv("all_shapes.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)
        
        df["prompt"] = instruction_tokens + "How many sides does the shape in the image have? Answer with the number." + end_tokens 

    elif args.task == 'two_shapes':
        df = pd.read_csv("two_shapes.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)
            
        df["prompt"] = instruction_tokens + df["prompt"] + end_tokens

    elif args.task == 'abstract':
        df = pd.read_csv("abstract_shapes.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)
        df["prompt"] = instruction_tokens + "How many sides does this shape have? Answer with one word." + end_tokens # Answer with one word.

    elif args.task == 'hept_ABC_123':
        df = pd.read_csv("heptagons_ABC_123.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)
            
        df["prompt"] = instruction_tokens + df["prompt"] + end_tokens
        
    elif args.task == 'triangle_cross_ABC_123':
        df = pd.read_csv("triangle_on_cross_ABC_123.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)    
        df["prompt"] = instruction_tokens + df["prompt"] + end_tokens
                

    elif args.task == 'mathverse_CoT':
        df = pd.read_csv("mathverse_revised.csv")
        if args.dataset_size == 'mini':
            df = df.sample(n=10, random_state=42)

        df["prompt"] = instruction_tokens + df["prompt"] + end_tokens

    df = generate_text_for_rows(
        df,
        processor=processor if args.model_version != 'internvl' else None,
        model=model,
        instruction_tokens=instruction_tokens,
        task=args.task,
        model_version=args.model_version,
        tokenizer=tokenizer if args.model_version == 'internvl' else None
    )

    output_file = f"{args.model_version}_{args.task}_{args.dataset_size}.csv"
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
