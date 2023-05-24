from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from starlette.responses import StreamingResponse

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    def results():
        for res, history_new in model.stream_chat(tokenizer,
                                                  prompt,
                                                  history=history,
                                                  max_length=max_length if max_length else 2048,
                                                  top_p=top_p if top_p else 0.7,
                                                  temperature=temperature if temperature else 0.95):
            yield 'event: message\ndata: {}\n\n'.format(res)
        yield 'event: message\ndata: [DONE]\n\n'

    torch_gc()
    return StreamingResponse(results(), media_type="text/event-stream")


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/root/models/int8", trust_remote_code=True)
    model = AutoModel.from_pretrained("/root/models/int8", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
