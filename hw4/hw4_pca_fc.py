import numpy as np
from skimage import io

def loadImages(folder):
	images = []
	for i in range(415): images.append(io.imread("%s/%d.jpg" %(folder, i)))
	images = np.array(images)
	return images.reshape(images.shape[0], 600 * 600 * 3)

def meanImages(images):
	return np.mean(images, axis=0)

def image_deprocess(image):
	image -= np.min(image)
	image /= np.max(image)
	image = (image * 255).astype(np.uint8)
	return image.reshape(600, 600, 3)

def saveImage(filename, image):
	print("\n------------Saving image: %s------------" %(filename))
	io.imsave(filename, image_deprocess(image))
	print("\n----------------------Done----------------------")

def showImage(image):
	io.imshow(image_deprocess(image))
	io.show()

def idxImage(filename):
	return int(filename[:len(filename) - 4])