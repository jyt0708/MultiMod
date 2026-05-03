from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import torch
torch.manual_seed(1234)

# pip uninstall -y optimum auto-gptq transformers
# pip install transformers==4.51.0
# pip install optimum==1.26.1
# pip install auto-gptq==0.7.1 --no-build-isolation
# pip install peft==0.7.1

QWEN_MODEL_DIR = "Qwen/Qwen-VL-Chat-Int4"

def init_qwen_vl(model_id=QWEN_MODEL_DIR):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    

    
    # 显式设置 pad_token, pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def get_prompts_from_image(image_path, model, tokenizer, user_instruction=""):
    """
    调用Qwen-VL生成JSON Prompt, 在Query中加入用户的额外要求。
    并基于图像主体属性 + user instruction 做冲突约束，自动推理 negative prompt。
    """
    if not user_instruction:
        task_description = "Identify the main subject in the image and describe its visual features. IGNORE the background."
    else:
        task_description = """The user wants to modify or repaint the main subject based on:'{user_instruction}'.
        You MUST:
        1. FOCUS ONLY on the main subject/object mentioned in the instruction.
        2. DO NOT describe the background, environment, or surrounding scenery.
        3. Identify the original attributes of the SUBJECT (color, shape, material, style) and compare these with the user's request.
        4. Do NOT ignore conflicts. Negative prompt must actively suppress unwanted original attributes.

        Example:
        Original Image: A black dog.
        User instruction: "A white cat."
        Output: {{
        "prompt": "a fluffy white cat",
        "negative_prompt": "black dog, puppy, black fur"
        }}
        """

    # 4. Move conflicting ORIGINAL subject attributes into the negative_prompt. If user wants to change 'A' to 'B', the negative prompt MUST include keywords describing 'A'
    
    system_prompt = f"""
    You are an expert Stable Diffusion XL prompt enginner specializing in SUBJECT-ONLY descriptions. 
    {task_description}

    Your goal is to generate a description that will be used for inpainting or subject replacement.

    Strictly follow this JSON format:
    {{
      "prompt": "Detailed positive prompt describing only the subject, and the user's instruction if present.",
      "negative_prompt": "Identify what's already in the image that must disapper, never repeat the user's request in the negative prompt."
    }}

    Return ONLY the JSON object.
    """

    query = tokenizer.from_list_format([
        {'image': image_path},  # 本地路径或url
        {'text': system_prompt}
    ])

    response, _ = model.chat(tokenizer, query=query, history=None)

    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL)
        if json_str:
            prompt_data = json.loads(json_str.group())
        else:
            raise ValueError("Response里不包含JSON")
    except Exception as e:
        print(f"JSON Parsing Error: {e}. Raw response: {response}")

        prompt_data = {
            "prompt": "masterpiece, highly detailed, " + response.replace('"', "'")[:200],
            "negative_prompt": ""
        }
    
    return prompt_data


# model, tokenizer = init_qwen_vl()
# prompt_data = get_prompts_from_image(r"D:\MM\pen.jpg", model, tokenizer, "futuristic, with glowing neon blue lines")
# print(prompt_data)

# {'prompt': 'A futuristic fountain pen', 'negative_prompt': 'Not a realistic fountain pen, not a photo, not monochrome, not black and white, not classical'}
#VLM 生成完成: {'prompt': 'A black Dachshund puppy sitting in a field of yellow flowers.', 'negative_prompt': 'color, style, material, lighting, objects'}