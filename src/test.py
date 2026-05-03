import requests
import os
import json
import time

def run_batch_test(tasks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "report.jsonl")
    vlm_url = "http://127.0.0.1:8001"
    sdxl_url = "http://127.0.0.1:8002/inpaint"

    for i, task in enumerate(tasks):
        img_path = task['image_path']
        user_inst = task['instruction']
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.png")

        print(f"\n--- [Task {i+1}/{len(tasks)}] ---")

        try:
            # 1. 向 VLM 服务请求 Prompt
            # vlm_resp = requests.post(f"{vlm_url}/get_prompt", 
            #                          json={"image_path": img_path, "instruction": user_inst})
            # prompt_data = vlm_resp.json()["data"]
            prompt_data = {"prompt": "A yellow, long-haired dachshund in a field of dandelions", "negative_prompt": "blurry, low quality"}
            print(f"VLM 生成完成: {prompt_data}")

            # 2. 向 SDXL 服务请求 Inpainting
            sd_resp = requests.post(sdxl_url, 
                                    json={
                                        "image_path": img_path,
                                        "prompt": prompt_data["prompt"],
                                        "negative_prompt": prompt_data["negative_prompt"],
                                        "output_path": output_path
                                    })
            
            if sd_resp.json()["status"] == "success":
                print(f"成功! 已保存至 {output_path}")

                # 3. 评估
                print("正在进行质量评估...")
                eval_resp = requests.post(f"{vlm_url}/evaluate", 
                                         json={"image_path": output_path, "instruction": prompt_data["prompt"]})
                eval_data = eval_resp.json()["data"]
                print(f"评估完成: {eval_data}")

                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "task_id": i + 1,
                    "instruction": user_inst,
                    "vlm_prompt": prompt_data["prompt"],
                    "vlm_negative_prompt": prompt_data["negative_prompt"],
                    "output_image": output_path,
                    "evaluation": eval_data
                }
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                print(f"SDXL 报错: {sd_resp.json()['message']}")

        except Exception as e:
            print(f"调度错误: {e}")

if __name__ == "__main__":
    test_tasks = [
        # {"image_path": r"D:\MM\pen.jpg", "instruction": "futuristic, with glowing neon blue lines"},   
        # {"image_path": r"D:\MM\fish.jpg", "instruction": "clownfish"},
        # {"image_path": r"D:\MM\woman.jpg", "instruction": "a woman with sunglasses"},
        # {"image_path": r"D:\MM\house.jpg", "instruction": "a stylish, luxurious house"},
        {"image_path": r"D:\MM\sea.jpg", "instruction": "a floating house on the sea"},
        {"image_path": r"D:\MM\dackel.jpg", "instruction": "yellow dachshund"},
        # {"image_path": r"D:\MM\dackel.jpg", "instruction": "long-hair dachshund"},
    ]
    run_batch_test(test_tasks, r"D:\MM\final_outputs_3")