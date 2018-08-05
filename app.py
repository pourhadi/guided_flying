# from __future__ import unicode_literals
import turicreate as tc
from prompt_toolkit import prompt
import threading
import socket
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from contextlib import contextmanager

HOST = ''
PORT = 9878
ADDR = (HOST,PORT)
BUFSIZE = 2048

image_model = tc.load_model('model.model')

targetSizeInInches = 5
distanceAtFull = 4
focalLength = 0

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def calculatePercentComplete(start, end, current):
    return (current - start) / (end - start)

def extrapolateValue(fromVal, toVal, percent):
    return fromVal + ((toVal - fromVal) * percent)

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

def main():
    img_path = prompt()
    img_path = img_path.replace(' ', '')
    image = tc.Image(img_path)
    run(image)

def run(image, removeAlpha = False):
    sf = tc.SFrame({'image': [image]})

    if (removeAlpha):
        sf['image'] = sf['image'].apply(lambda x: drop_alpha(x))

    # print("in run")
    with suppress_stdout():
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


def mainVideo():

    cv2.namedWindow("preview")
    cv2.namedWindow("result")

    vc = cv2.VideoCapture(0)

    height = 0
    width = 0

    currentFrame = 0

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        
        if (currentFrame == 10):
            currentFrame = 0
            image = tc.Image(_image_data=frame.tobytes(),
                    _width=width,
                    _height=height,
                    _channels=3,
                    _format_enum=2,
                    _image_data_size=width * height * 3)

            thread = threading.Thread(target=run, args=([image, False]))
            thread.start()
        else:
            currentFrame = currentFrame + 1

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def mainServer():
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    serv.bind(ADDR)
    serv.listen()

    while True:
        conn, addr = serv.accept()

        while True:
            data = conn.recv(BUFSIZE)
            if not data: break
            try:
                height = 480
                width = 640
                print(len(data))
                image = tc.Image(_image_data=data,
                        _width=width,
                        _height=height,
                        _channels=3,
                        _format_enum=2,
                        _image_data_size=width * height * 3)

                thread = threading.Thread(target=run, args=([image]))
                thread.start()
            except:
                print("error")
    conn.close()

class Watcher:
    DIRECTORY_TO_WATCH = "/Users/danielpourhadi/tmp/guided/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()

        self.observer.join()
        
class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        try:
            if event.is_directory:
                return None

            elif event.event_type == 'modified':

            # Taken any action here when a file is modified.

                image = tc.Image('/Users/danielpourhadi/tmp/guided/image.jpg')
                thread = threading.Thread(target=run, args=([image]))
                thread.start()

        except:
            print('error')

def mainWatcher():
    w = Watcher()
    w.run()

if __name__ == '__main__':
    mainWatcher()
    # mainServer()
    # mainVideo()
    # while (True):
    #     main()

