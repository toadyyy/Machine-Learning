# You can copy parts of this code into your jupyter notebook or python code as needed :)
import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.io

# Firstly, you can use the skimage.io.imload method to load an image:
img1 = skimage.io.imread('UHG.png')
img2 = skimage.io.imread('CITEC.png')
img3 = skimage.io.imread('UBI_X.png')

# To display an image, simply use the plt.imshow method:
plt.figure() # <- This line is not strictly required in jupyter notebooks

plt.title('The University Main Building')
plt.imshow(img1)
plt.axis('off') # <- This line prevents the x- and y-Axes from being shown

plt.show() # <- This line is not strictly required in jupyter notebooks

# If we want, we can also add multiple images in one plot:
images = np.array([img1, img2, img3])
image_labels = ['University Building', 'CITEC Building', 'X Building']

fig, ax = plt.subplots(1, 3, figsize=(16, 3))
for i, axi in enumerate(ax.flat):
    axi.set_title(image_labels[i])
    axi.imshow(images[i])
    axi.set(xticks=[], yticks=[])  # <- This line prevents the x- and y-Axes from being shown


plt.show() # <- This line is not strictly required in jupyter notebooks
