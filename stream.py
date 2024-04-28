# import libraries
import cv2

from vidgear.gears import CamGear
stream = CamGear(source='https://youtube.com/live/g1tb--rpfVE', stream_mode = True, logging=True).start() # YouTube Video URL as input

# infinite loop
while True:
    
    frame = stream.read()
    # read frames

    # check if frame is None
    if frame is None:
        #if True break the infinite loop
        break
    
    # do something with frame here
    
    cv2.imshow("Output Frame", frame)
    # Show output window

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()