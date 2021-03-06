import sys, os
import datetime as dt
import time
from subprocess import Popen
from collect_video import collect_video_chunks
import collect_video
import helper_functions as hf
import picamera
   
def get_players():
    print " "
    players = 2*[None]
    players[0] = raw_input("Enter player 1 (player who breaks) initials: ")
    players[1] = raw_input("Enter player 2 initials: ")
    return players

camera = picamera.PiCamera(resolution=(1640, 922))

# MAIN INDEFINITE LOOP
while True:

    os.system('clear')
    print "=============================================="
    print "BALS (BILLIARDS ANALYTICS AND LEARNING SYSTEM)"
    print "             START SCREEEN"
    print "=============================================="
    print " "
    
    print "Press 's' to begin a game.  Press 'q' to kill the program."
    
    # Start Loop
    while True:

        keypress = hf.getch()
            
        # If the 'q' key is pressed, quit
        if keypress == "q":
            quit()
        
        # If the 's' key is pressed, start
        elif keypress == "s":
            break

    print("Let's begin!")

    players = get_players()
    print " "
    gameStartTime = dt.datetime.now()
    print players[0].upper() + " versus " + players[1].upper() + "  " + gameStartTime.strftime('%Y-%m-%d %H:%M:%S')
        
    print " "
    print "Please rack and begin your 8-ball game.  Good luck!"
    print ""
    print "When game is done, please press 'ESC'."
    print ""
    print "Recording segments ..."
    

    # In game
    filename = collect_video_chunks(camera)

    gameEndTime = dt.datetime.now()
    gameDuration = ((gameEndTime - gameStartTime).seconds)/60.0
    print ""
    print ""
    print "Let's handle some record-keeping please ..."
    print players[0].upper() + " = player 1"
    print players[1].upper() + " = player 2"
    stripes = int(raw_input("Stripes were played by 1 or 2? ")) - 1
    winner  = int(raw_input("Who won, 1 or 2? (Enter 0 if no winner) ")) - 1
    print ""
    print "Summary:"
    print players[0].upper() + " versus " + players[1].upper()
    print "Game duration (min): %.1f" %  gameDuration
    print players[stripes].upper() + " was stripes."
    stripesPlayer = players[stripes].upper()
    if stripes == 0:
        print players[1].upper() + " was solids."
        solidsPlayer = players[1].upper()
    elif stripes == 1:
        print players[0].upper() + " was solids."
        solidsPlayer = players[0].upper()
    else:
        print "Invalid entry."
        
    if winner == -1:
        print "No winner."
    elif winner == 0:
        print players[0].upper() + " was the winner!"
        winnerPlayer = players[0].upper()
    elif winner == 1:
        print players[1].upper() + " was the winner!"
        winnerPlayer = players[1].upper()
    else:
        print "Invalid entry."
                
    try:
        with open(filename + "_meta.txt", "w") as text_file:
            text_file.write("Player 1: %s\n" % players[0].upper())
            text_file.write("Player 2: %s\n" % players[1].upper())
            text_file.write("Winner:   %s\n" % winnerPlayer)
            text_file.write("Solids:   %s\n" % solidsPlayer)
            text_file.write("Stripes:  %s\n" % stripesPlayer) 
            text_file.write("Duration: %f\n" % gameDuration)
    except:
        print "Meta fail."

#    try:
#        proc = Popen('sudo mv -fv *.h264 remoteVideoStorage', shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
#        print "Video files moved to remote storage."
#    except:
#        print "Video files FAILED to be moved to remote storage", (sys.exc_info()[0])
    
    time.sleep(5)
