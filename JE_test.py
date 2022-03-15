# %%


from skimage import io, filters
import os

set_ID = '11059585'
photo_ID = '0'

conc_ID = set_ID + '/' + set_ID + '_' + photo_ID + '.jpg'

path = 'data/uob_image_set/'

filename = os.path.join(path,conc_ID)
image = io.imread(filename)

# ... or any other NumPy array!
mejeiring = filters.meijering(image)
edges = filters.sobel(image)

io.imshow(mejeiring)
io.show()

io.imshow(edges)
io.show()

