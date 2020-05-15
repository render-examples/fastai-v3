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

export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_name = 'export.pkl'

classes = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani',
 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird',
 'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting',
 'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee',
 'Chuck will Widow', 'Brandt Cormorant', 'Red faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird',
 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo',
 'Gray crowned Rosy Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher',
 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch',
 'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe', 'Western Grebe', 'Blue Grosbeak',
 'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull',
 'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring billed Gull', 'Slaty backed Gull', 'Western Gull',
 'Anna Hummingbird', 'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 'Pomarine Jaeger',
 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird',
 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake',
 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red breasted Merganser',
 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole',
 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood Pewee',
 'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven', 'White necked Raven',
 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black throated Sparrow',
 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow',
 'Lincoln Sparrow', 'Nelson Sharp tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow',
 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White crowned Sparrow', 'White throated Sparrow', 'Cape Glossy Starling',
 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager',
 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern',
 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black capped Vireo', 'Blue headed Vireo', 'Philadelphia Vireo', 'Red eyed Vireo',
 'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler',
 'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler',
 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut sided Warbler', 'Golden winged Warbler',
 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler',
 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler',
 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing',
 'American Three toed Woodpecker', 'Pileated Woodpecker', 'Red bellied Woodpecker', 'Red cockaded Woodpecker',
 'Red headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren',
 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']
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
    #await download_file(export_file_url, path / export_file_name)
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
