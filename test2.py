import os
import os.path as osp
import cv2 as cv
import numpy as np
import tensorflow as tf

from skimage.metrics import structural_similarity as compare_ssim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
node_names = {'input_image':      'placeholder/input_image:0',
              'input_mask':       'placeholder/input_mask:0',
              'output_attention': 'concat_1:0',
              'output_image':     'Tanh:0'}
data_root = 'data'
output_dir = 'output'


def read_image(image_path, channels):
	if channels == 3:
		image = cv.imread(image_path, cv.IMREAD_COLOR)
	else:
		image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
		image = np.expand_dims(image, axis=2)
	image = image.astype(np.float) / 127.5 - 1.0
	return image


def test():
	if not osp.exists(data_root):
		print("No data directory")
		exit(0)

	model_pb = osp.join('model', 'model.pb')
	if not osp.exists(model_pb):
		print("No pre-trained model")
		exit(0)

	if not osp.exists(output_dir):
		os.makedirs(output_dir)

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.compat.v1.Session(config=config) as sess:

		with tf.io.gfile.GFile(model_pb, 'rb') as f:
			graph_def = tf.compat.v1.GraphDef()
			graph_def.ParseFromString(f.read())
			sess.graph.as_default()
			tf.import_graph_def(graph_def, name='')

		sess.run(tf.compat.v1.global_variables_initializer())

		input_image      = sess.graph.get_tensor_by_name(node_names['input_image'])
		input_mask       = sess.graph.get_tensor_by_name(node_names['input_mask'])
		output_attention = sess.graph.get_tensor_by_name(node_names['output_attention'])
		output_image     = sess.graph.get_tensor_by_name(node_names['output_image'])

		image_list  = sorted(os.listdir(osp.join(data_root, 'noshadow')))
		image_batch = np.zeros((1, 256, 256, 3), dtype=np.float)
		mask_batch  = np.zeros((1, 256, 256, 1), dtype=np.float)

		for i in image_list:
			print(i)

			image_batch[0] = read_image(osp.join(data_root, 'noshadow', i), 3)
			mask_batch[0]  = 0 - read_image(osp.join(data_root, 'mask', i), 1)

			feed_dict = {input_image: image_batch,
			             input_mask:  mask_batch}

			image, attention = sess.run([output_image, output_attention], feed_dict=feed_dict)
			image = ((1.0 + image) * 127.5).astype(np.uint8)
			object_attention = (attention[0, :, :, 0] * 255.0).astype(np.uint8)
			shadow_attention = (attention[0, :, :, 1] * 255.0).astype(np.uint8)

			cv.imwrite(os.path.join(output_dir, i), image[0])
			cv.imwrite(os.path.join(output_dir, 'object_' + i), object_attention)
			cv.imwrite(os.path.join(output_dir, 'shadow_' + i), shadow_attention)

			# custom code
			g1 = cv.cvtColor(image[0], cv.COLOR_BGR2GRAY)
			g2 = cv.cvtColor(((1.0 + image_batch) * 127.5).astype(np.uint8)[0], cv.COLOR_BGR2GRAY)
			(score, diff) = compare_ssim(g1, g2, full=True)
			diff = (diff * 255).astype("uint8")
			print("SSIM for image {}: {}".format(i, score))
			cv.fastNlMeansDenoising(diff, diff, 42)
			cv.imwrite(os.path.join(output_dir, 'diff_' + i), diff)
			ret, thresh = cv.threshold((255 - diff), 127.5, 255, 0)
			M = cv.moments(thresh)
			x = int(M["m10"] / M["m00"])
			y = int(M["m01"] / M["m00"])
			print(x, y)


if __name__ == '__main__':
	test()
