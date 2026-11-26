"""Join independent preparation and image upload with bounded cancellation."""
import asyncio
import time


async def join_preparation_upload(preparation, upload):
    started = time.perf_counter()
    times = {}

    async def timed(name, work):
        result = await work
        times[name] = (time.perf_counter() - started) * 1000
        return result

    tasks = [asyncio.create_task(timed('preparation_ms', preparation)),
             asyncio.create_task(timed('upload_ms', upload))]
    try:
        prepared, _ = await asyncio.gather(*tasks)
        times['joined_ms'] = (time.perf_counter() - started) * 1000
        return prepared, times
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
