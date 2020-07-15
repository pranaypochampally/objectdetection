import numpy as np 
from yolov3 import YOLOv3Net
from yolov3 import  parse_cfg

def load_weights(model, cfgfile, weightfile):
	fp = open(weightfile, 'rb')
	np.fromfile(fp, dtype=np.int32, count=5)

	blocks = parse_cfg(cfgfile)

	for i, block in enumerate(blocks[1:]):
		if block["type"] == "convolutional":

			conv_layer = model.get_layer('conv_'+str(i))
			print("layer: ",i+1, conv_layer)

			filters = conv_layer.filters
			k_size = conv_layer.kernel_size[0]
			in_dim = conv_layer.input_shape[-1]

			if "batch_normalize" in block:

				norm_layer = model.get_layer('bnorm_'+str(i))

				print("layer: ",i+1, norm_layer)

				bn_weights = np.fromfile(fp, dtype=np.float32, count= 4*filters)
				bn_weights = bn_weights.reshape((4,filters))[[1, 0, 2, 3]]

			else:
				conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)


			conv_shape = (filters, in_dim, k_size, k_size)

			conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))

			conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])


			if "batch_normalize" in block:
				norm_layer.set_weights(bn_weights)
				conv_layer.set_weights([conv_weights])
			else:
				conv_layer.set_weights([conv_weights, conv_bias])

		#assert len(fp.read()) == 0, 'failed to read data'
	fp.close()

def main():
	weightfile = "weights/yolov3.weights"
	cfgfile = "cfg/yolov3.cfg"

	model_size = (416, 416, 3)
	num_classes = 80

	model = YOLOv3Net(cfgfile, model_size, num_classes)

	load_weights(model, cfgfile, weightfile)

	try:
		model.save_weights('weights/yolov3_weights.tf')
		print('\nFile is saved in tensor  format')
	except  IOError:
		print(" caouldn't write file to tensor")
if __name__ == "__main__":
	main()