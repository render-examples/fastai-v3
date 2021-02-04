import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *


#I needed to add these imports in order to avoid the below error (Path not defined)
import fastbook
from fastbook import *
from fastai.vision.widgets import *

# RUN python app/server.py
# Feb 4 10:01:31 AM  #10 0.464 Traceback (most recent call last):
# Feb 4 10:01:31 AM  #10 0.464   File "app/server.py", line 17, in <module>
# Feb 4 10:01:31 AM  #10 0.464     path = Path(__file__).parent
# Feb 4 10:01:31 AM  #10 0.464 NameError: name 'Path' is not defined

from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/s/pwu7pztq2e55era/KombuchaVision_2.4.21.pkl?dl=1'
export_file_name = 'export.pkl'

classes = ['Mold', 'Not Mold']
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
        learn = load_learner(path / export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise
            
# Traceback (most recent call last):
# Feb 4 10:05:02 AM  #10 5.638   File "app/server.py", line 59, in <module>
# Feb 4 10:05:02 AM  #10 5.638     learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
# Feb 4 10:05:02 AM  #10 5.638   File "/usr/local/lib/python3.8/asyncio/base_events.py", line 616, in run_until_complete
# Feb 4 10:05:02 AM  #10 5.638     return future.result()
# Feb 4 10:05:02 AM  #10 5.638   File "app/server.py", line 46, in setup_learner
# Feb 4 10:05:02 AM  #10 5.638     learn = load_learner(path, export_file_name)
# Feb 4 10:05:02 AM  #10 5.638   File "/usr/local/lib/python3.8/site-packages/fastai/learner.py", line 539, in load_learner
# Feb 4 10:05:02 AM  #10 5.638     res = torch.load(fname, map_location='cpu' if cpu else None)
# Feb 4 10:05:02 AM  #10 5.638   File "/usr/local/lib/python3.8/site-packages/torch/serialization.py", line 581, in load
# Feb 4 10:05:02 AM  #10 5.638     with _open_file_like(f, 'rb') as opened_file:
# Feb 4 10:05:02 AM  #10 5.638   File "/usr/local/lib/python3.8/site-packages/torch/serialization.py", line 230, in _open_file_like
# Feb 4 10:05:02 AM  #10 5.638     return _open_file(name_or_buffer, mode)
# Feb 4 10:05:02 AM  #10 5.638   File "/usr/local/lib/python3.8/site-packages/torch/serialization.py", line 211, in __init__
# Feb 4 10:05:02 AM  #10 5.638     super(_open_file, self).__init__(open(name, mode))
# Feb 4 10:05:02 AM  #10 5.638 IsADirectoryError: [Errno 21] Is a directory: 'app'


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
