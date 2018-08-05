import turicreate as tc
from PIL import Image, ImageDraw
import sys

image_dir = sys.argv[1]

image = tc.Image(image_dir)

model = tc.load_model('model.model')

test = tc.SFrame({'image': [image]})
test['predictions'] = model.predict(test)

test['image_with_predictions'] = tc.object_detector.util.draw_bounding_boxes(test['image'], test['predictions'])
test[['image', 'image_with_predictions']].explore()

test['image_with_predictions'][0].save('test.png')