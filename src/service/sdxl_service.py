import torch
import uvicorn
import sys
import asyncio

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
# from gpu_scheduler import submit_gpu_task
from gpu_lock import gpu_lock
from starlette.concurrency import run_in_threadpool
from sdxl_pipeline.smart_composition import SDXLControlNetInpainter

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
from u2net.model.u2net import U2NET


app = FastAPI()

# Global Queue: 不允许多个请求同时进GPU
# gpu_queue = Queue()


# 加载 u2net 和 Inpainter
def load_detector(weight=r"D:\MM\MultiMod\u2net.pth"):
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(weight, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    net.eval()
    if torch.cuda.is_available(): net.cuda()
    return net

net = load_detector()
inpainter = SDXLControlNetInpainter(
    u2net_model=net,    
    controlnet_id="diffusers/controlnet-canny-sdxl-1.0",
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0" 
)

class SDXLRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str
    output_path: str


def run_inpaint(req: SDXLRequest):
    print("[GPU LOCK] Waiting SDXL...")
    with gpu_lock:
        print("[GPU LOCK] SDXL Enter")
        assets = inpainter.prepare_assets(req.image_path, target_size=1024)
        result_img = inpainter.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            assets=assets,
            steps=30,
            control_scale=0.3
        )

        result_img.save(req.output_path)
    
        torch.cuda.empty_cache()
        
        print("[GPU LOCK] SDXL Exit")

    return {
        "status": "success",
        "saved_at": req.output_path
    }

# async def gpu_worker():
#     while True:
#         # 取任务
#         req, future = await gpu_queue.get()
#         try:
#             print(f"[QUEUE] Start job: {req.output_path}")

#             # 放线程里执行，避免阻塞event loop
#             result = await asyncio.to_thread(
#                 run_inpaint,
#                 req
#             )

#             future.set_result(result)

#             print(f"[QUEUE] Finished job: {req.output_path}")
            
#         except Exception as e:
#             future.set_result({
#                 "status": "error",
#                 "message": str(e)
#             })
#             print(f"[QUEUE] Failed: {e}")

#         finally:
#             gpu_queue.task_done()



# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(gpu_worker())


@app.post("/inpaint")
async def do_inpaint(req: SDXLRequest):
    # run_inpaint不是async，event loop会堵塞（GPU推理会卡住API）
    # 所以丢进线程池跑
    result = await run_in_threadpool(run_inpaint, req)
    return result


# @app.post("/inpaint")
# def do_inpaint(req: SDXLRequest):
#     try:
#         assets = inpainter.prepare_assets(req.image_path, target_size=1024)
#         result_img = inpainter.generate(
#             prompt=req.prompt,
#             negative_prompt=req.negative_prompt,
#             assets=assets,
#             steps=30,
#             control_scale=0.3
#         )
#         result_img.save(req.output_path)
#         torch.cuda.empty_cache()
#         return {"status": "success", "saved_at": req.output_path}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)