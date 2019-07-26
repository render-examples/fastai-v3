import aiohttp
import asyncio
import uvicorn
import os
from fastai import *
from fastai.vision import *
import random
import string
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from google.cloud import storage

export_file_url = 'https://www.dropbox.com/s/yuwyshs6tmwp46b/trained_model_1%20%281%29.pkl?raw=1'
export_file_name = 'export.pkl'

classes = ['Cardboard', 'E-Waste', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/feedback')
async def feedback(request):
    html_file = path / 'view' / 'feedback.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

@app.route('/submit', methods=['POST'])
async def submit(request):
    img_data = (await request.form())
    for key in img_data.keys():
        pred = key
    img_bytes = await (img_data[pred].read())
    img = open_image(BytesIO(img_bytes))
    prediction = pred
    # bucket upload...
    # """Uploads a file to the bucket."""
    storage_client = storage.Client.from_service_account_json(
        'app/AMLI-6588677dc859.json')
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket('amli_trashnet_photos')
    alias = ''.join(random.choice(string.ascii_letters) for _ in range(32))
    blobstr = 'Web_data/' + prediction + '/' + alias
    blob = bucket.blob(blobstr)
    blob.upload_from_string(img_bytes, content_type = 'image/jpeg')
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
