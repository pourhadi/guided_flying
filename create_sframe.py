
import os
import io
import xml.etree.ElementTree as ET
from lxml import etree
from object_detection.utils import dataset_util
import tensorflow as tf

import turicreate as tc

def drop_alpha(image):
    return tc.Image(_image_data=image.pixel_data[..., :3].tobytes(),
                    _width=image.width,
                    _height=image.height,
                    _channels=3,
                    _format_enum=2,
                    _image_data_size=image.width * image.height * 3)


images_dir = 'images'
annotations_dir = 'annotations'

sf_images = tc.image_analysis.load_images(images_dir, recursive=False, random_order=True)
sf_images['image'] = sf_images['image'].apply(lambda x: drop_alpha(x))
sf_images['name'] = sf_images['path'].apply(lambda path: os.path.basename(path))

sf_images['label'] = 'target'

annotations = {'name': [], 'annotations': []}

for filename in os.listdir(images_dir):
    if (filename == '.DS_Store'): continue
    
    name = os.path.splitext(filename)[0]
    labels_path = annotations_dir + '/' + name + '.xml'

    xml_str = open(labels_path, 'r').read()

    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])

    box_width = xmax - xmin
    box_height = ymax - ymin

    x = xmin + (box_width / 2)
    y = ymin + (box_height / 2)

    annotations['name'].append(name + '.png')
    annotations['annotations'].append([{'coordinates': {'height': box_height, 'width': box_width, 'x': x, 'y': y}, 'label': 'target'}])


sf_labels = tc.SFrame(annotations)

sf = sf_images.join(sf_labels, on='name', how='left')

sf.save('images.sframe')
