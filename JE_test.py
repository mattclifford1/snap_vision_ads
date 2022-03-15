# %%


from skimage import io, filters
import os

image = io.imread('data/uob_image_set/11059585/11059585_0.jpg')

# ... or any other NumPy array!
mejeiring = filters.meijering(image)
edges = filters.sobel(image)

io.imshow(mejeiring)
io.show()

io.imshow(edges)
io.show()

