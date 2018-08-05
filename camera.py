import cv2
import socket
import threading

HOST = 'localhost'
PORT = 9878
ADDR = (HOST,PORT)
BUFSIZE = 2048


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

height = 0
width = 0

currentFrame = 0

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(ADDR)

def send(data):
    client.send(data.tobytes())

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    print(width)
    print(height)
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    print(len(frame.tobytes()))

    # thread = threading.Thread(target=send, args=([frame]))
    # thread.start()

    if (currentFrame == 5):
        cv2.imwrite('/Users/danielpourhadi/tmp/guided/image.jpg', frame)
        currentFrame = 0
    else:
        currentFrame += 1
        
    key = cv2.waitKey(20)
        
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")




client.close()