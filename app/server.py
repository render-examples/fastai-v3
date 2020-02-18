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

#export_file_url = 'https://www.dropbox.com/s/e8bwv831u7zoapg/clouds.pkl?raw=1'
#export_file_name = 'clouds.pkl'

export_file_url = 'https://www.dropbox.com/s/0umzro4cy7web2q/clouds864.pkl?raw=1'
export_file_name = 'clouds864.pkl'


classes = ['Altocumulus', 'Altocumulus Lenticularis', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Cumulonimbus', 'Cumulus', 'Mammatus', 'Nimbostratus', 'Stratocumulus', 'Stratus']
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
    # prediction = learn.predict(img)[0]
    result = learn.predict(img)
    prediction = result[0]
    classIndex = result[1].item()
    classProb = result[2][classIndex].item()
    classPercent = round(classProb*100)
    output = str(prediction) + str(' : ') + str(classPercent) + str('% confidence') 
    #return JSONResponse({'result' : output})
    
    _,_,losses = learn.predict(img)
    sortedClasses = sorted(zip(classes, map(float, losses)),key=lambda p: p[1],reverse=True)
    
    output = str(' ')
    # Don't output more than 8 types, and stop when less than 2% rounded
    
    for i in range(7):
        if sortedClasses[i][1] > 0.015:
            if i>0:
                output = output + str(', ')
            output = output + str(sortedClasses[i][0]) + str(' ') + str(round(100*sortedClasses[i][1])) + str('%')
        else:
            break
          
    return JSONResponse({'result' : output })
    
    
    #output1 = str(sortedClasses[0][0]) + str(' ') + str(round(100*sortedClasses[0][1])) + str('%')
    #output2 = str(sortedClasses[1][0]) + str(' ') + str(round(100*sortedClasses[1][1])) + str('%')
    #output3 = str(sortedClasses[2][0]) + str(' ') + str(round(100*sortedClasses[2][1])) + str('%')
    
    #return JSONResponse({'result' : [output1,output2,output3] })
    
    #return JSONResponse({ 'result' : sorted(zip(classes, map(float, losses)),key=lambda p: p[1],reverse=True) [:3] })
 



if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
