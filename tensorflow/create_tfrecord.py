
import hashlib
import os
import io
import xml.etree.ElementTree as ET
import tensorflow as tf
from lxml import etree

from object_detection.utils import dataset_util
import PIL.Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_dir', '', 'Path to directory of images')
flags.DEFINE_string('labels_dir', '', 'Path to directory of labels')
FLAGS = flags.FLAGS


def create_tf_example(example):

    image_path = os.getcwd() + '/' +  FLAGS.images_dir + example
    labels_path = os.getcwd() + '/' +  FLAGS.labels_dir + os.path.splitext(example)[0] + '.xml'

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_img = fid.read()
    encoded_io = io.BytesIO(encoded_img)
    image = PIL.Image.open(encoded_io)

    key = hashlib.sha256(encoded_img).hexdigest()

    with tf.gfile.GFile(labels_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    # Read the image
    # img = Image.open(image_path)
    width = int(data['size']['width'])
    height = int(data['size']['height'])


    image_format = 'png'

    # Read the label XML
    # tree = ET.parse(labels_path)
    # root = tree.getroot()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])

    xmins.append(xmin / width)
    ymins.append(ymin / height)
    xmaxs.append(xmax / width)
    ymaxs.append(ymax / height)

    classes_text = ['target'.encode('utf8')]
    classes = [1]

    print(xmins)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_img),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for filename in os.listdir(FLAGS.images_dir):
        if (filename == '.DS_Store'): continue
        tf_example = create_tf_example(filename)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
