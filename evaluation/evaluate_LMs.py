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

def remove_example(text):
    return re.sub(r'### Example Output:.*?\nIMPORTANT:', 'IMPORTANT:', text, flags=re.DOTALL)

def remove_awow(prompt):
    return re.sub(r'\s*Answer with one word\.\s*$', '', prompt)

def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def evaluate_gpt(prompt, model_version):
    API_KEY = "YOUR API KEY HERE"

    client = openai.OpenAI(api_key=API_KEY)

    response = client.chat.completions.create(
        model=model_version,  
        messages=[
            {"role": "system", "content": "Answer the user's prompt based on your knowledge."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
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

        if model_version == 'llava-1.5' or model_version == 'llava-1.6':
            prompt = row['text_only_prompt']
            inputs = processor.tokenizer(text=prompt, padding=True, return_tensors="pt").to(model.device)
            outputs = model.language_model.generate(**inputs, max_new_tokens=500)

            if model_version == 'llava-1.5':
                predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1]
            elif model_version == 'llava-1.6':
                predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_answer = clean_instruction_tokens(predicted_answer)

        elif model_version == 'internvl':
            generation_config = dict(max_new_tokens=500, do_sample=False)
            question = row["text_only_prompt"]
            pixel_values=None
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            predicted_answer = response

        elif model_version == "gpt-4o" or model_version == "gpt-4-turbo": 
            prompt = row["text_only_prompt"]
            predicted_answer = evaluate_gpt(prompt, model_version)
            
        elif model_version == 'qwen':
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["text_only_prompt"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                return_tensors="pt",
            ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
            
            outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
            
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("\nassistant\n")[-1]
            
        elif model_version == 'llava-one':
            conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": row['text_only_prompt']}
                        ],
                    },
                ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                    text=prompt,
                    return_tensors="pt",
                ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
            outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0] #.split("assistant\n\n")[-1]

        
        elif model_version == 'llama-3.2':
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row['text_only_prompt']}
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                    text=prompt,
                    return_tensors="pt",
                ).to(model.device, torch.float16 if model.device.type == "cuda" else torch.float32)
            
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("assistant\n\n")[-1]


        elif model_version == 'molmo':
            print(model)
            
            text_input = row["text_only_prompt"]  # The text prompt you want to process
            
            # Tokenize the input text
            inputs = processor.tokenizer(text_input, return_tensors="pt")
            
            # Move tensors to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            input_ids = inputs["input_ids"]
            seq_length = input_ids.shape[1]
            
            position_ids = torch.arange(seq_length, dtype=torch.long, device=model.device).unsqueeze(0)
            
            # Forward pass through the language model
            with torch.no_grad():
                output = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,  # Ensure position IDs are correctly passed
                )
            
            # Extract only the **newly generated** tokens (not the input)
            predicted_ids = output.logits[:, seq_length-1:, :].argmax(dim=-1)  # Get next tokens
            
            # Decode only the newly predicted tokens
            predicted_answer = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            print(predicted_answer)

            
        elif model_version == 'janus':
            prompt = f"\n<|User|>: {remove_awow(row['text_only_prompt'])} \n<|Assistant|>:"
            input_ids = processor.tokenizer(prompt, return_tensors='pt').to(model.device) 

            output = model.language_model.generate(**input_ids,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=512,)
             
            predicted_answer = processor.tokenizer.decode(output[0].cpu().tolist(), skip_special_tokens=True)

        elif model_version in ['math-llava', 'g-llava']:
            prompt = row["text_only_prompt"]
            
            if task == "counting":
                prompt = prompt.split("\n\n### Example")[0]
            
            args = argparse.Namespace(
                query=prompt,  # Modify this with your actual prompt
                conv_mode=None,  # Keep as None to use the auto-detected mode
                temperature=0,  
                top_p=1,  
                num_beams=1,  
                max_new_tokens=1000,  
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
            
            predicted_answer = evalmodel_LM(args, model_name, tokenizer, model, image_processor, context_len)
            
        elif model_version == 'math-puma':
            system_prompt = ""
            text_only = True
            conv = []
            if system_prompt:
                conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

            user_content = [{"type": "text", "text": row['text_only_prompt']}] 

            conv.append({"role": "user", "content": user_content})
            prompt = processor.apply_chat_template(conv, add_generation_prompt=True)

            # Process inputs
            inputs = processor(
                text=prompt,
                images=raw_image if not text_only else None,
                return_tensors="pt"
            ).to(model.device).to(torch.float16)

            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs, 
                                        max_new_tokens=100, 
                                        do_sample=False,
                                        num_beams=1)

            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split("\nassistant\n")[-1]
        
        generated_texts.append(predicted_answer)

    df['generated_text'] = generated_texts
    return df

def main():
    parser = argparse.ArgumentParser(description="Run LLAVA, Qwen, and InternVL models for shape identification or counting tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-1.5', 'llava-1.6', 'blip', 'qwen', 'internvl', 'gpt-4o', 'gpt-4-turbo','llava-one','llama-3.2', 'molmo', 'math-llava', 'g-llava', 'math-puma', 'janus'], required=True, help="Choose the model version (1.5, 1.6, qwen, or internvl).")
    parser.add_argument('--task', type=str, choices=['shape_id', 'sides_id'], required=True, help="Choose the task (shape_id or sides_id).")
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
        instruction_tokens = "[INST]\n"
        end_tokens = "[/INST]"
        
    elif args.model_version == 'llava-1.5':
        processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        model = LlavaForConditionalGeneration.from_pretrained(
            'llava-hf/llava-1.5-7b-hf', quantization_config=bnb_config
        ).to(0)
        instruction_tokens = "USER:\n"
        end_tokens = "\nASSISTANT:"

    elif args.model_version == 'blip':
        processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        instruction_tokens = ""
        end_tokens = ""
    elif args.model_version == "gpt-4o" or args.model_version == "gpt-4-turbo": 
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
        instruction_tokens = ""
        end_tokens = ""
        
    elif args.model_version == 'llava-one':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True
        )
        instruction_tokens = ""
        end_tokens = ""
        
    elif args.model_version == 'llama-3.2':
        processor = AutoProcessor.from_pretrained("ruslanmv/Llama-3.2-11B-Vision-Instruct")
        model = MllamaForConditionalGeneration.from_pretrained(
            "ruslanmv/Llama-3.2-11B-Vision-Instruct", torch_dtype=torch.float16, quantization_config=bnb_config, low_cpu_mem_usage=True
        )
        instruction_tokens = ""
        end_tokens = ""


    elif args.model_version in ['math-llava', 'g-llava']:
        if args.model_version == 'math-llava':
            model_path = "Zhiqiang007/Math-LLaVA"
        
        elif args.model_version == 'g-llava':
            model_path = "renjiepi/G-LLaVA-7B"
        
        model_base = None  # If fine-tuning, change this
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
        
        text_only = ["What is the standard name of a 7 sided regular polygon? Answer with one word.",
                     "What is the standard name of a 6 sided regular polygon? Answer with one word.",
                     "What is the standard name of an 8 sided regular polygon? Answer with one word.",
                     "What is the standard name of a 5 sided regular polygon? Answer with one word.",
                     "What is the standard name of a 4 sided regular polygon? Answer with one word.", 
                     "What is the standard name of a 3 sided regular polygon? Answer with one word."]

        #the following models struggle to understand the question sometimes, so the prompt may need to be tweaked. For example, they will say "tetrahedron" instead of "square". 
        if args.model_version in ['janus', 'llava-1.5', 'g-llava', 'math-llava']:
            text_only = [
                "What is the name of a regular polygon that has 7 sides? Answer with one word.",
                "What is the name of a regular polygon that has 6 sides? Answer with one word.",
                "What is the name of a regular polygon that has 8 sides? Answer with one word.",
                "What is the name of a regular polygon that has 5 sides? Answer with one word.",
                "What is the name of a regular polygon that has 4 sides? Answer with one word.",
                "What is the name of a regular polygon that has 3 sides? Answer with one word."]
            

        df = pd.DataFrame(index=range(6))
        df["text_only_prompt"] = text_only
        df["text_only_prompt"] = instruction_tokens + df["text_only_prompt"] + end_tokens

    elif args.task == 'sides_id':
        
        text_only = [
                "How many sides does an octagon have? Answer with one word.",
                "How many sides does a heptagon have? Answer with one word.",
                "How many sides does a hexagon have? Answer with one word.",
                "How many sides does a pentagon have? Answer with one word.",
                "How many sides does a square have? Answer with one word.",
                "How many sides does a triangle have? Answer with one word."]
        
        df = pd.DataFrame(index=range(6))
        df["text_only_prompt"] = text_only
        df["text_only_prompt"] = instruction_tokens + df["text_only_prompt"] + end_tokens
        
    
    df = generate_text_for_rows(
        df,
        processor=processor if args.model_version != 'internvl' else None,
        model=model,
        instruction_tokens=instruction_tokens,
        task=args.task,
        model_version=args.model_version,
        tokenizer=tokenizer if args.model_version == 'internvl' else None
    )

    output_file = f"{args.model_version}_{args.task}_{args.dataset_size}_text_only.csv"
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
