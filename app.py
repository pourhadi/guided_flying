# from __future__ import unicode_literals
import turicreate as tc
import threading
import time
import os

import subprocess as sp
import numpy

HOST = ''
PORT = 9878
ADDR = (HOST,PORT)
BUFSIZE = 2048

image_model = tc.load_model('model.model')

targetSizeInInches = 5
distanceAtFull = 4
focalLength = 0


def calculatePercentComplete(start, end, current):
    return (current - start) / (end - start)

def extrapolateValue(fromVal, toVal, percent):
    return fromVal + ((toVal - fromVal) * percent)

def createImage(numpy_image, width, height):
    return tc.Image(_image_data=numpy_image.tobytes(),
                    _width=width,
                    _height=height,
                    _channels=3,
                    _format_enum=2,
                    _image_data_size=width * height * 3)

def drop_alpha(image):
    return tc.Image(_image_data=image.pixel_data[..., :3].tobytes(),
                    _width=image.width,
                    _height=image.height,
                    _channels=3,
                    _format_enum=2,
                    _image_data_size=image.width * image.height * 3)

def calculateHorizontal(x, box_width, image_width):
    center = x + (box_width / 2)
    percent = calculatePercentComplete(0, image_width, center)
    h = extrapolateValue(-1, 1, percent)
    return h

def calculateVertical(y, box_height, image_height):
    center = y + (box_height / 2)
    percent = calculatePercentComplete(image_height, 0, center)
    v = extrapolateValue(-1, 1, percent)
    return v

def calculateDistance(box_height, image_height):
    focalLength = (image_height * distanceAtFull) / targetSizeInInches
    return (targetSizeInInches * focalLength) / box_height

def run(image, removeAlpha = False):
    sf = tc.SFrame({'image': [image]})

    if (removeAlpha):
        sf['image'] = sf['image'].apply(lambda x: drop_alpha(x))

    # print("in run")

    predictions = image_model.predict(sf)
    # print(predictions)
    if (len(predictions) < 1 or len(predictions[0]) < 1):
        return

    # sf['image_with_predictions'] = tc.object_detector.util.draw_bounding_boxes(sf['image'], predictions)
    # cv2.imshow("result", sf['image_with_predictions'][0])
    x = predictions[0][0]['coordinates']['x']
    y = predictions[0][0]['coordinates']['y']
    width = predictions[0][0]['coordinates']['width']
    height = predictions[0][0]['coordinates']['height']
    horizontal = calculateHorizontal(x, width, image.width)
    vertical = calculateVertical(y, height, image.height)
    distance = calculateDistance(height, image.height)
    print('[' + str(horizontal) + "," + str(vertical) + "," + str(distance) + ']')


# # TARGET_DIR = os.path.join(os.getcwd(), 'target')
# TARGET_DIR = '/private/tmp/drone-vision'
# class Watcher:
#     DIRECTORY_TO_WATCH = TARGET_DIR

#     def __init__(self):
#         self.observer = Observer()

#     def run(self):
#         event_handler = Handler()
#         self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
#         self.observer.start()
#         try:
#             while True:
#                 time.sleep(5)
#         except:
#             print('stopped because exception')
#             self.observer.stop()

#         self.observer.join()
        
# class Handler(FileSystemEventHandler):

#     @staticmethod
#     def on_any_event(event):
#         try:
#             if event.is_directory:
#                 return None

#             elif event.event_type == 'modified':
#             # Taken any action here when a file is modified.

#                 image = tc.Image(os.path.join(TARGET_DIR, 'capture.jpg'))
#                 thread = threading.Thread(target=run, args=([image]))
#                 thread.start()

#         except:
#             print('error')

# def mainWatcher():
#     w = Watcher()
#     w.run()
# 960x720

def mainPipeReader():
    width = 960
    height = 720
    FFMPEG_BIN = "ffmpeg"
    command = [ FFMPEG_BIN,
            '-i', '/app/target/drone_vision_pipe',             # fifo is the named pipe
            '-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
            '-vcodec', 'rawvideo',
            '-an','-sn',              # we want to disable audio processing (there is no audio)
            '-f', 'image2pipe', '-']   

    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

    frame = 0
    while True:
        raw_image = pipe.stdout.read(width*height*3)
        if (frame < 10):
            frame += 1
            pipe.stdout.flush()
            continue
        
        frame = 0
        # Capture frame-by-frame
        # transform the byte read into a numpy array
        image =  numpy.frombuffer(raw_image, dtype=numpy.uint8)
        image = image.reshape((height,width,3))          # Notice how height is specified first and then width
        if image is not None:
            simage = createImage(image, width, height)
            thread = threading.Thread(target=run, args=([simage]))
            thread.start()

        pipe.stdout.flush()

if __name__ == '__main__':
    mainPipeReader()
    # mainServer()
    # mainVideo()
    # while (True):
    #     main()

