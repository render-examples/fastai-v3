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

# export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_url = 'https://www.googleapis.com/drive/v3/files/1YMgds-ERUKRERWXy1sNkcP-wPAT6qrDR?alt=media&key=AIzaSyBDcJM7FN_vag_lQBanpzkFluTgauKYQMk'

export_file_name = 'my_model.pkl'

classes = ['acer_platanoides', 'acer_saccharinum', 'aesculus_flava', 'ailanthus_altissima', 'amelanchier_canadensis', 'betula_alleghaniensis', 'betula_nigra', 'carpinus_betulus', 'castanea_dentata', 'catalpa_speciosa', 'chamaecyparis_thyoides', 'chionanthus_retusus', 'cornus_florida', 'evodia_daniellii', 'ficus_carica', 'fraxinus_nigra', 'fraxinus_pennsylvanica', 'ilex_opaca', 'juglans_cinerea', 'juniperus_virginiana', 'maclura_pomifera', 'magnolia_acuminata', 'magnolia_soulangiana', 'magnolia_tripetala', 'malus_angustifolia', 'malus_coronaria', 'malus_pumila', 'nyssa_sylvatica', 'oxydendrum_arboreum', 'phellodendron_amurense', 'picea_orientalis', 'picea_pungens', 'pinus_densiflora', 'pinus_echinata', 'pinus_parviflora', 'pinus_sylvestris', 'populus_grandidentata', 'prunus_serrulata', 'quercus_falcata', 'quercus_macrocarpa', 'quercus_marilandica', 'quercus_michauxii', 'quercus_montana', 'quercus_muehlenbergii', 'quercus_phellos', 'quercus_stellata', 'quercus_virginiana', 'salix_babylonica', 'stewartia_pseudocamellia', 'styrax_obassia', 'taxodium_distichum', 'ulmus_glabra']
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
