import os
import argparse
import numpy as np
import tensorflow as tf

from skimage.transform import resize
from imageio import imread, imsave
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/E2_2.png',
                    help='input image paths.')

# colored map
# floorplan_map = {
# 	0: [255,255,255], # background: white
# 	1: [192,192,224], # closet: purple
# 	2: [192,255,255], # batchroom/washroom: light blue
# 	3: [224,255,192], # livingroom/kitchen/dining room: light green
# 	4: [255,224,128], # bedroom: yellow
# 	5: [255,160, 96], # hall: orange
# 	6: [255,224,224], # balcony: light pink
# 	7: [255,255,255], # not used
# 	8: [255,255,255], # not used
# 	9: [255, 60,128], # door & window: bright pink
# 	10:[  0,  0,  0]  # wall: black
# }

# binary map
floorplan_map = {
	0: [255,255,255], # background
	1: [255,255,255], # closet
	2: [255,255,255], # batchroom/washroom
	3: [255,255,255], # livingroom/kitchen/dining room
	4: [255,255,255], # bedroom
	5: [255,255,255], # hall
	6: [255,255,255], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255,255,255], # door & window
	10:[  0,  0,  0]  # wall
}

####### DOORS AND WINDOWS ########

# floorplan_map = {
# 	0: [255,255,255], # background: white
# 	1: [255,255,255], # closet: purple
# 	2: [255,255,255], # batchroom/washroom: light blue
# 	3: [255,255,255], # livingroom/kitchen/dining room: light green
# 	4: [255,255,255], # bedroom: yellow
# 	5: [255,255,255], # hall: orange
# 	6: [255,255,255], # balcony: light pink
# 	7: [255,255,255], # not used
# 	8: [255,255,255], # not used
# 	9: [255, 60,128], # door & window: bright pink
# 	10:[255,255,255]  # wall: black
# }

##################################

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im


def process_image_file(im_path,intermediate_path):
	# load input
	#im = imread(args.im_path, mode='RGB')
	im = imread(im_path, mode='RGB')
	im = im.astype(np.float32)
	im = resize(im, (512, 512, 3)) / 255.

	# create tensorflow session
	with tf.compat.v1.Session() as sess:
		# initialize
		sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
						  tf.compat.v1.local_variables_initializer()))

		# restore pretrained model
		saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		# get default graph
		graph = tf.compat.v1.get_default_graph()

		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')
		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit], \
											  feed_dict={x: im.reshape(1, 512, 512, 3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		floorplan[room_boundary == 1] = 9
		floorplan[room_boundary == 2] = 10
		floorplan_rgb = ind2rgb(floorplan)

		# plot results
		#plt.subplot(121)
		#plt.imshow(im)
		#plt.subplot(122)
		plt.imshow(floorplan_rgb / 255.)
		#plt.show()
		plt.axis('off')
		plt.savefig(intermediate_path, bbox_inches='tight', pad_inches=0)

def process_img(im_path, intermediate_path, mode):
	print("demoNEWALG.py -->  process_imageNEWALG :  ")
	print(im_path)
	print(intermediate_path)
	process_image_file(im_path, intermediate_path)