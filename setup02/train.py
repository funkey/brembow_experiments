from brembow import Cage
from brembow import PointSpreadFunction, GaussianPSF
from brembow.gp import SimulateCages
from funlib.learn.torch.models import UNet, ConvPass
import gunpowder as gp
import numpy as np
import torch
import zarr
import logging
from gunpowder.profiling import Timing, TimingSummary, ProfilingStats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# cages and render parameters
cage1 = Cage("../data/example_cage", 1)
psf = GaussianPSF(intensity=0.125, sigma=(1.0, 1.0))
min_density = 1e-6
max_density = 1e-6

# create model, loss, and optimizer
unet = UNet(
    in_channels=1,
    num_fmaps=24,  # this needs to be increased later (24)
    fmap_inc_factor=3,  # this needs to be increased later (3)
    downsample_factors=[[1, 2, 2], [1, 2, 2], [1, 2, 2],],
    kernel_size_down=[[[3, 3, 3], [3, 3, 3]]]*3,
    kernel_size_up=[[[3, 3, 3], [3, 3, 3]]]*2,
    padding='valid')
model = torch.nn.Sequential(
    unet,
    ConvPass(24, 1, [(1, 1, 1)], activation='Sigmoid'))
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# declare gunpowder arrays
raw = gp.ArrayKey('RAW')
seg = gp.ArrayKey('SEGMENTATION')
out_cage_map = gp.ArrayKey('OUT_CAGE_MAP')
out_density_map = gp.ArrayKey('OUT_DENSITY_MAP')
prediction = gp.ArrayKey('PREDICTION')


class PrepareTrainingData(gp.BatchFilter):

    def process(self, batch, request):

        batch[out_cage_map].data = batch[out_cage_map].data.astype(np.float32)
        batch[out_cage_map].spec.dtype = np.float32


# assemble pipeline
sourceA = gp.ZarrSource(
    '../data/cropped_sample_A.zarr',
    {raw: 'raw', seg: 'segmentation'},
    {
        raw: gp.ArraySpec(interpolatable=True),
        seg: gp.ArraySpec(interpolatable=False)
    }
)
sourceB = gp.ZarrSource(
    '../data/cropped_sample_B.zarr',
    {raw: 'raw', seg: 'segmentation'},
    {
        raw: gp.ArraySpec(interpolatable=True),
        seg: gp.ArraySpec(interpolatable=False)
    }
)
sourceC = gp.ZarrSouce(
    '../data/cropped_sample_C.zarr',
    {raw: 'raw', seg: 'segmentation'},
    {
        raw: gp.ArraySpec(interpolatable=True),
        seg: gp.ArraySpec(interpolatable=False)
    }
)

source = (sourceA, sourceB, sourceC) + gp.MergeProvider()
print(source)
normalize = gp.Normalize(raw)
simulate_cages = SimulateCages(
    raw,
    seg,
    out_cage_map,
    out_density_map,
    psf,
    (min_density, max_density),
    [cage1],
    0.5)
add_channel_dim = gp.Stack(1)
stack = gp.Stack(5)
prepare_data = PrepareTrainingData()
train = gp.torch.Train(
    model,
    loss,
    optimizer,
    inputs={
        'input': raw
    },
    loss_inputs={
        0: prediction,
        1: out_cage_map
    },
    outputs={
        0: prediction
    })
pipeline = (
    source +
    normalize +
    gp.RandomLocation() +
    simulate_cages +
    add_channel_dim +
    stack +
    prepare_data +
    gp.PreCache(num_workers=40)+
    train +
    gp.PrintProfilingStats(every=1))


print("PIPELINE")
print(pipeline)

# compute a valid input and output size

# for x and y:
#
# 124 -> 120                88 -> 84
#         |                  |
#         60 -> 56    48 -> 44
#                |     |
#               28 -> 24
#
# for z:
#
# 28 -> 24                12 -> 8
#       |                 |
#       24 -> 20    16 -> 12
#             |     |
#             20 -> 16

# 124 ⇒ 84 works for x and y
# 28 ⇒ 8 works for z

with gp.build(source):
    voxel_size = source.spec[raw].voxel_size

request = gp.BatchRequest()

input_size = gp.Coordinate((28, 124, 124)) * voxel_size
output_size = gp.Coordinate((8, 84, 84)) * voxel_size

request.add(raw, input_size)
request.add(out_cage_map, output_size)
request.add(out_density_map, output_size)
request.add(seg, output_size)
request.add(prediction, output_size)

def render_into_zarr(batch):
    datafile = 'testV'+str(batch.iteration)+'.zarr'
    print(f"Rendering {datafile} into zarr...")
    batch[raw].data = np.squeeze(batch[raw].data)
    batch[raw].data = batch[raw].data[0,:,:,:]
    batch[out_cage_map].data = np.squeeze(batch[out_cage_map].data)
    batch[out_cage_map].data = batch[out_cage_map].data[0,:,:,:]
    batch[out_density_map].data = np.squeeze(batch[out_density_map].data)
    batch[out_density_map].data = batch[out_density_map].data[0,:,:,:]
    batch[seg].data = np.squeeze(batch[seg].data)
    batch[seg].data = batch[seg].data[0,:,:,:]
    batch[prediction].data = np.squeeze(batch[prediction].data)
    batch[prediction].data = batch[prediction].data[0,:,:,:]
    with zarr.open(datafile, 'w') as f:
        f['render'] = batch[raw].data
        f['render'].attrs['resolution'] = batch[raw].spec.voxel_size
        f['cage_map'] = batch[out_cage_map].data
        f['cage_map'].attrs['resolution'] = batch[out_cage_map].spec.voxel_size
        f['cage_map'].attrs['offset'] = batch[out_cage_map].spec.roi.get_begin()
        f['density_map'] = batch[out_density_map].data
        f['density_map'].attrs['resolution'] = batch[out_density_map].spec.voxel_size
        f['density_map'].attrs['offset'] = batch[out_density_map].spec.roi.get_begin()
        f['seg'] = batch[seg].data
        f['seg'].attrs['resolution'] = batch[seg].spec.voxel_size
        f['seg'].attrs['offset'] = batch[seg].spec.roi.get_begin()
        f['prediction'] = batch[prediction].data
        f['prediction'].attrs['resolution'] = batch[prediction].spec.voxel_size
        f['prediction'].attrs['offset'] = batch[prediction].spec.roi.get_begin()

with gp.build(pipeline):
    for i in range(5000):
        batch = pipeline.request_batch(request)
        with open('loss.txt', 'a') as file:
            file.write(str(batch.iteration) + ", "  + str(batch.loss) + "\n")
            file.flush()
        if(i%500 == 0):
            render_into_zarr(batch)

print("done")
