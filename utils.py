import tensorflow as tf 
import  numpy as np 
import cv2

def non_max_suppression(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
	bbox, confs, class_probs = tf.split(inputs, [4, 1,-1], axis=-1)
	bbox = bbox/model_size[0]

	scores = confs*class_probs

	boxes, scores, classes, valid_detection = tf.image.combined_non_max_suppression(
		boxes = tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
		scores = tf.reshape(scores,(tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
		max_output_size_per_class = max_output_size_per_class,
		max_total_size = max_output_size,
		iou_threshold = iou_threshold,
		score_threshold = confidence_threshold
		)
	return boxes, scores, classes, valid_detection

def resize_image(inputs, model_size):
	inputs = tf.image.resize(inputs, model_size)
	return inputs

def load_class_names(file_name):
	with open(file_name, 'r') as f:
		class_names = f.read().splitlines()

	return class_names

def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
	center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1,1,1,1,1,-1], axis=-1)

	x1 = center_x-width/2.0
	y1 = center_y-height/2.0
	x2 = center_x+width/2.0
	y2 = center_y+height/2.0

	inputs = tf.concat([x1,y1,x2,y2,confidence,classes], axis=-1)

	boxes_dicts = non_max_suppression(inputs, model_size,max_output_size,max_output_size_per_class,iou_threshold,confidence_threshold)

	return boxes_dicts

def draw_outputs(img, boxes, objectness, classes, nums, class_names):

	boxes,objectness,classes,nums = boxes[0],objectness[0],classes[0],nums[0]

	boxes = np.array(boxes)

	for i in range(nums):
		x1y1 = tuple((boxes[i,0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))

		x2y2 = tuple((boxes[i,2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))

		img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)

		img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2)

		return img
