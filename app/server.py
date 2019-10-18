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

classes = ['acer_platanoides', 'acer_saccharinum', 'aesculus_flava', 'ailanthus_altissima', 'amelanchier_canadensis',
           'betula_alleghaniensis', 'betula_nigra', 'carpinus_betulus', 'castanea_dentata', 'catalpa_speciosa',
           'chamaecyparis_thyoides', 'chionanthus_retusus', 'cornus_florida', 'evodia_daniellii', 'ficus_carica',
           'fraxinus_nigra', 'fraxinus_pennsylvanica', 'ilex_opaca', 'juglans_cinerea', 'juniperus_virginiana',
           'maclura_pomifera', 'magnolia_acuminata', 'magnolia_soulangiana', 'magnolia_tripetala', 'malus_angustifolia',
           'malus_coronaria', 'malus_pumila', 'nyssa_sylvatica', 'oxydendrum_arboreum', 'phellodendron_amurense',
           'picea_orientalis', 'picea_pungens', 'pinus_densiflora', 'pinus_echinata', 'pinus_parviflora',
           'pinus_sylvestris', 'populus_grandidentata', 'prunus_serrulata', 'quercus_falcata', 'quercus_macrocarpa',
           'quercus_marilandica', 'quercus_michauxii', 'quercus_montana', 'quercus_muehlenbergii', 'quercus_phellos',
           'quercus_stellata', 'quercus_virginiana', 'salix_babylonica', 'stewartia_pseudocamellia', 'styrax_obassia',
           'taxodium_distichum', 'ulmus_glabra']
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

species_details = {
    'acer_platanoides': 'also',
    # 'acer_platanoides': "also known as Norway maple, used for furniture, flooring and musical instruments. begins flowering in spring. The flowers emerge during April, but in some of the coolest areas of the tree's range they may not appear until June. More : https://en.wikipedia.org/wiki/acer_platanoides",
    "acer_saccharinum": "commonly known as silver maple,[4] creek maple, silverleaf maple,[4] soft maple, large maple,[4] water maple,[4] swamp maple,[4] or white maple. seeds are also a food source for squirrels, chipmunks and birds. The bark can be eaten by beaver and deer. produced before the leaves in early spring,[6] with the seeds maturing in early summer. More: https://en.wikipedia.org/wiki/acer_saccharinum",
    "aesculus_flava": "commonly known as the yellow buckeye, common buckeye, or sweet buckeye, is cultivated as an ornamental tree, The flowers are produced in panicles in spring. More - https://en.wikipedia.org/wiki/Aesculus_flava",
    "ailanthus_altissima": "known as tree of heaven, ailanthus, varnish tree. appear from mid-April in the south of its range to July in the north. addition to its use as an ornamental plant, the tree of heaven is also used for its wood and as a host plant to feed silkworms",
    "amelanchier_canadensis": "(bilberry,[1] Canadian serviceberry, chuckleberry, currant-tree,[2] juneberry, shadblow serviceberry, shadblow, shadbush, shadbush serviceberry, sugarplum, thicket serviceberry). It is a deciduous shrub or small tree growing to 0.5–8 m tall. It is used as a medicinal plant,[6] food, and ornamental plant.[7] It is sometimes made into bonsai",
    "betula_alleghaniensis": "the yellow birch[1] or golden birch. prefers to grow in cooler conditions and is often found on north facing slopes, swamps, stream banks, and rich woods. most important species of birch for lumber and is the most important hardwood lumber tree in eastern Canada. In the past, yellow birch has been used for distilling wood alcohol, acetate of lime and for tar and oils. More https://en.wikipedia.org/wiki/Betula_alleghaniensis",
    "betula_nigra": "the black birch, river birch or water birch. is a deciduous tree growing to 25–30 meters (80–100 ft) with a trunk 50 to 150 centimeters (20 to 60 in) in diameter. making it a favored ornamental tree for landscape use. is not typically used in the commercial lumber industry, due to knotting, but its strong, closely grained wood is sometimes used for local furniture, woodenware, and fuel.[4][8] This species is utilized by many local bird species, such as waterfowl, ruffed grouse, and wild turkey. More https://en.wikipedia.org/wiki/Betula_nigra",
    "carpinus_betulus": "commonly known as the European or common hornbeam. requires a warm climate for good growth, and occurs only at elevations up to 600 metres (1,969 ft). cultivated as an ornamental tree, for planting in gardens and parks throughout north west Europe. More https://en.wikipedia.org/wiki/Carpinus_betulus",
    "castanea_dentata": "known as American chestnut. rapidly growing deciduous hardwood tree, historically reaching up to 30 metres (98 ft) in height, and 3 metres (9.8 ft) in diameter. Chestnuts are edible raw or roasted, though typically preferred roasted. Native Americans used various parts of the American chestnut to treat ailments such as whooping cough, heart conditions and chafed skin.[2]. It is superior in quality to any found in Europe. The wood is straight-grained, strong, and easy to saw and split, and it lacks the radial end grain found on most other hardwoods, good for furnitures. More https://en.wikipedia.org/wiki/American_chestnut",
    "catalpa_speciosa": "commonly known as the northern catalpa, hardy catalpa, western catalpa, cigar tree, catawba-tree, or bois chavanon. medium-sized, deciduous tree growing to 15–30 meters tall and 12 meters wid. widely planted as an ornamental tree. It is adapted to moist, high pH soil and full sun. More https://en.wikipedia.org/wiki/Catalpa_speciosa",
    "chamaecyparis_thyoides": "commonly knowns as Atlantic white cedar, Atlantic white cypress, southern white cedar, whitecedar, or false-cypress. is of some importance in horticulture, with several cultivars of varying crown shape, growth rates and foliage color having been selected for garden planting, most common use of white cedar wood is lumber, for which stands usually require 70 years of growth from germination to harvest. The lumber may be used in house construction.More https://en.wikipedia.org/wiki/Chamaecyparis_thyoides",
    "chionanthus_retusus": "the Chinese fringetree,[2] is a flowering plant in the family Oleaceae. is a deciduous shrub or small to medium-sized tree growing to 20 metres (70 ft) in height, with thick, fissured bark. cultivated in Europe and North America as an ornamental tree, valued for its feathery white flowerheads.",
    "cornus_florida": "the flowering dogwood, is a species of flowering tree in the family Cornaceae. tree is commonly planted as an ornamental in residential and public areas because of its showy bracts and interesting bark structure. Flowering dogwood does best horticulturally in moist, acidic soil in a site with some afternoon shade, but good morning sun.",
    "evodia_daniellii": "a small to medium sized deciduous tree, typical landscape size is 25' to 30'; but it can get larger, generally as wide as tall, most often found with a short main trunk that divides into several main branches. ",
    "ficus_carica": "",
    "fraxinus_nigra": "",
    "fraxinus_pennsylvanica": "",
    "ilex_opaca": "",
    "juglans_cinerea": "",
    "juniperus_virginiana": "",
    "maclura_pomifera": "",
    "magnolia_acuminata": "",
    "magnolia_soulangiana": "",
    "magnolia_tripetala": "",
    "malus_angustifolia": "",
    "malus_coronaria": "",
    "malus_pumila": "",

}


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
    try:
        value = species_details[str(prediction)]
    except:
        value = None

    return JSONResponse({'result': value, 'details': str(value)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
