import picamera
import helper_functions as hf
import datetime as dt
from subprocess import Popen
import sys, select, termios, tty

# Home:
# sudo sshfs smckenna@192.168.1.18:/home/smckenna/piVideo remoteVideoStorage/

# Work:
# sudo sshfs smckenna@192.168.1.112:/home/smckenna/BALS/piVideo remoteVideoStorage/

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def grabStill(camera):
    camera.capture("still.jpg", use_video_port=True)
 
def moveStill():
    proc = Popen('sudo cp -f still.jpg remoteVideoStorage', shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)



old_settings = termios.tcgetattr(sys.stdin)

   
def collect_video_chunks(camera):
    
#    camera = picamera.PiCamera(resolution=(1640, 922))
    camera.framerate = 30
    camera.rotation = 0
    camera.clock_mode = "reset"

    chunkSize = 60 # seconds
#    camera.annotate_frame_num  = True
#    camera.annotate_background = picamera.Color('black')
#    camera.annotate_foreground = picamera.Color('white')
    filename = dt.datetime.now().strftime('BALS_%Y-%m-%d-%H%M')

    try:
        tty.setcbreak(sys.stdin.fileno())

        camera.start_recording('%s_1.h264' % filename, quality=20)
        camera.wait_recording(chunkSize)
        print "Video " + filename + "_1.h264 saved locally."
        i = 2
        start = dt.datetime.now()
        now = start

        while True:
        
#            camera.annotate_frame_num  = True
            if (dt.datetime.now() - start).seconds >= chunkSize:
                camera.split_recording('%s_%d.h264' % (filename, i), quality=20)
                print "Video " + filename + "_" + str(i) + ".h264 saved locally."
                i = i + 1
                start = dt.datetime.now()
                print "Elapsed time: " + str((start-now).total_seconds())
                now = start
            camera.wait_recording(0.2)
#            camera.annotate_frame_num = True

            if isData():
                c = sys.stdin.read(1)
                if c == '\x1b':         # x1b is ESC
                    camera.split_recording('%s_%d.h264' % (filename, i), quality=20)
                    print "Video " + filename + "_" + str(i) + ".h264 saved locally."
                    i = i + 1
                    start = dt.datetime.now()
                    print "Elapsed time: " + str((start-now).total_seconds())
                    grabStill(camera)
                    moveStill()
                    camera.stop_recording()
                    break

            if i > 30:
                print "Maximum game time reached.  Exiting recording mode."
                camera.split_recording('%s_%d.h264' % (filename, i), quality=20)
                print "Video " + filename + "_" + str(i) + ".h264 saved locally."
                i = i + 1
                start = dt.datetime.now()
                print "Elapsed time: " + str((start-now).total_seconds())
                grabStill(camera)
                moveStill()
                camera.stop_recording()
                break

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    return filename 

# avconv -r 30 -i 1.h264 -vcodec copy 1.mp4
# avconv -r 30 -i 1.h264 -c:v libx264 1.mp4
