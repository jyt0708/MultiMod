from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(1234)

QWEN_MODEL_DIR = "Qwen/Qwen-VL-Chat-Int4"

def init_qwen_vl(model_id=QWEN_MODEL_DIR):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    

    
    # 显式设置 pad_token, pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


