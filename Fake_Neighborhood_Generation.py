import shapefile, random, math, pickle
from shapely.geometry import Polygon, Point
from pathlib import Path
from ast import literal_eval
from collections import defaultdict
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

DATA_DIR = Path('/media/alex/Backup/Alex Yelp/csvs')
use_idf = True

# Polygon generation code from https://stackoverflow.com/a/25276331

def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points

def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x

def generate_real_fake_values(fakes_to_generate):
    d = {}
    for shape in sf.iterShapeRecords():
        d[shape.record.AREA_NAME] = Polygon(shape.shape.points)
        
    allcoords = []
    for cdict in bf.coordinates.values:
        cdict = literal_eval(cdict)
        if cdict['longitude'] is not None and cdict['latitude'] is not None:
            allcoords.append((cdict['longitude'],cdict['latitude']))
            
    areas = []
    for t in d.values():
        areas.append(t.area)
        
    mu = np.mean(areas)
    si = np.std(areas)
    
    factor = 1e9
    fakes = []
    for i in range(fakes_to_generate):
        coord = random.choice(allcoords)
        fakes.append(
            generatePolygon(
            ctrX=coord[0]*factor,
            ctrY=coord[1]*factor,
            aveRadius=random.gauss(mu,si)*factor,
            irregularity=random.uniform(0,1),
            spikeyness=random.uniform(0,0.5),
            numVerts=random.randint(3,50)
            )
        )
        
    for i,fake in enumerate(fakes):
        for j,points in enumerate(fake):
            fake[j] = (points[0]/factor,points[1]/factor)
        fakes[i] = fake

    polygon_fakes = [Polygon(fake) for fake in fakes]

    n2bid_fakes = defaultdict(list)
    for bid,loc in (bf[['id','coordinates']].values):
        loc = literal_eval(loc)
        if loc['longitude'] is not None and loc['latitude'] is not None:
            p = Point(loc['longitude'],loc['latitude'])
            for k,v in enumerate(polygon_fakes):
                if v.contains(p):
                    n2bid_fakes[k].append(bid)

    neighbourhoodcounts_fakes = {}
    bid_index = list(bf_text.business.values)
    for k,v in n2bid_fakes.items():
        total = np.zeros((bf_vecs.shape[1],))
        for bid in v:
            if bid in bids:
                total += bf_vecs[bid_index.index(bid),:]
        neighbourhoodcounts_fakes[k] = np.asarray(total)[0]

    names_fakes = list(neighbourhoodcounts_fakes.keys())
    toremove = []
    for k,v in neighbourhoodcounts_fakes.items():
        if np.sum(v) != 0:
            neighbourhoodcounts_fakes[k] /= np.sum(v)  
        else:
            toremove.append(k)
    for k in toremove:
        neighbourhoodcounts_fakes.pop(k)
    return names_fakes,np.asarray(list(neighbourhoodcounts_fakes.values()))

bf = pd.read_csv(Path.joinpath(DATA_DIR,'businesses.csv'))
bf_text = pd.read_csv(Path.joinpath(DATA_DIR,'business_text_stripped_all.csv')).dropna()
bids = list(bf_text['business'].values)
sf = shapefile.Reader("/home/alex/uem_projects/neighbourhoods_wgs84/NEIGHBORHOODS_WGS84")

p = Path('models/vectorizer.pickle')
with open(p,'rb') as f:
    vectorizer = pickle.load(f)
words = vectorizer.get_feature_names()

if use_idf:
    p = Path('models/tfidftransformer.pickle')
    with open(p,'rb') as f:
        transformer = pickle.load(f)

transform = lambda x: normalize(
    transformer.transform(
        vectorizer.transform(
            x.text.values
        )
    ),
    norm='l1',
    axis=1)
bf_vecs = transform(bf_text)

generate_real_fake_values(140)