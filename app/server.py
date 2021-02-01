import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1QIbLP5c3Hwd5WyAKk8JOBUjc6mfOrYsL'
#export_file_url = 'https://secure-web.cisco.com/174QUIBcNPtGWG8cunZ-4rk0BGU8xhEbuxBmdbDIb6pFzljxpwD8B5_T-W72G8k1c5ienRz5egv1Lvl6rkxme6qtiCzVKw23H5SaPZ2DZeKExa-bk9dgZ-h5WcvLy3lJ2JiKm8XNBrEEoDEj_xTiaP7uHmWbYiRw9AOtvnMGQQZYUhU3Tbr0H1rKD4D2UDJN17WQQb1eMSMLxEeUMdanWRVvxCF1FYOLW4Olm6qzzomlpcUCTG8ORuVLqwDOx7_z1JSZxTCFxjkBNKXUwvAv-Rb6Vg6JShd1idZc7KX74O8bldJeGWHKIWe0EaFnxAGZfuPIlVxpnYjVa3mC6kjtKm75bvXCzHp5kLbCzo1UWckg9Os2RiRPCL-Sa300YWt5ln4v4RtfOi-GoOeq87glOg8eL0-HNqmNTa2UMq6UB3XFPRuWDGzx9tSVRyTChbUoxEwQglZeGDPetmg-il3EJbw6F9qr5OXM4MnL8vNJbdRjn3GTdt48_0UuijsntUBW3yFs4KeXToWX0tqfQKpDTAQ/https%3A%2F%2Fwww.dropbox.com%2Fl%2Fscl%2FAACCW_Y_-ZT8PCDgqKTrucWMOuXOFlioEgc'
export_file_name = 'KombuchaVision_1.22.21.pkl'

classes = ['Not Mold', 'Mold']
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
