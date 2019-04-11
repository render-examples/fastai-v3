from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://www.hongkonghustle.com/dimsum.pkl'
export_file_name = 'dimsum.pkl'

classes = ['baked_crispy_bun_with_bbq_pork', 'baked_pork_bun_glazed', 'beef_ball', 'beef_tripe', 'char_siu_bao', 'char_siu_cheung', 'cheung_fan', 'chicken_feet', 'coconut_pudding', 'curry_squid', 'dja_leung', 'egg_tart', 'fish_ball', 'fu_pei_guen', 'fun_gor', 'ha_cheung', 'ham_sui_gok', 'har_gow', 'honey_comb_tripe', 'lo_bak_go', 'lo_mai_gai', 'ma_lai_gao', 'mango_pudding', 'nai_wong_bao', 'osmanthus_jelly', 'pai_gwat', 'pan_fried_pork_bun', 'pork_puffs', 'potstickers', 'quails_egg_siumai', 'radish_puff', 'scallop_dumpling', 'sesame_ball', 'shark_fun_dumpling', 'siu_mai', 'spring_roll', 'stuffed_eggplant', 'tang_yuan_dessert', 'three_treasures', 'water_chestnut_cake', 'wu_gok', 'xiao_long_bao']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
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
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
