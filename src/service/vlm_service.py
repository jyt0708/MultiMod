import uvicorn
import torch
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from qwen_vl.model import init_qwen_vl
from gpu_scheduler import submit_gpu_task, start_gpu_worker
from qwen_vl.prompt_construction import get_prompts_from_image


app = FastAPI()

# 全局变量加载模型，避免重复加载
model, tokenizer = init_qwen_vl()

class VLMRequest(BaseModel):
    image_path: str
    instruction: str


def run_vlm(req: VLMRequest):
    prompt_data = get_prompts_from_image(
        req.image_path, 
        model, 
        tokenizer, 
        req.instruction
    )
        
    torch.cuda.empty_cache()
    return {"status": "success", "data": prompt_data}
     
     
# FastAPI启动时，后台启动一个永久GPU worker，只启动一次
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_gpu_worker())
    
    
@app.post("/get_prompt")
async def generate_prompt(req: VLMRequest):

    result = await submit_gpu_task(
        run_vlm,
        req
    )

    return result
     

# @app.post("/get_prompt")
# async def generate_prompt(req: VLMRequest):
#     try:
#         prompt_data = get_prompts_from_image(req.image_path, model, tokenizer, req.instruction)
#         return {"status": "success", "data": prompt_data}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


@app.post("/evaluate")
async def evaluate_image(req: VLMRequest):
    query = f"User original instruction: {req.instruction}. \
              Please evaluate the quality of this image based on the instruction. \
              Give a score (1-10) and a brief reason. Format: Score: X; Reason: Y."
    
    query_text = tokenizer.from_list_format([
        {'image': req.image_path},
        {'text': query},
    ])
    
    response, _ = model.chat(tokenizer, query=query_text, history=None)
    
    return {"status": "success", "data": response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
    # python -m service.vlm_service