import asyncio
from asyncio import Queue

# 所有GPU任务（vlm+sdxl）共享同一个Queue
gpu_queue = Queue()


async def start_gpu_worker():
    while True:
        # 取任务
        func, args, future = await gpu_queue.get()

        try:
            print(f"[GPU] Start: {func.__name__}")
            
            # 放线程里执行，避免阻塞event loop
            result = await asyncio.to_thread(
                func,
                *args
            )
            future.set_result(result)
            print(f"[GPU] Done: {func.__name__}")

        except Exception as e:
            future.set_result({
                "status": "error",
                "message": str(e)
            })
            print(f"[GPU] Error: {e}")

        finally:
            gpu_queue.task_done()


async def submit_gpu_task(func, *args):
    loop = asyncio.get_running_loop()
    
    # 占位
    future = loop.create_future()
    
    await gpu_queue.put(
        (func, args, future)
    )
    print(f"[GPU] Queue size: {gpu_queue.qsize()}")

    # 等待gpu worker完成
    return await future