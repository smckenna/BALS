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
    stripes = raw_input("Did " + players[0].upper() + " play stripes (y/n)? ")
    winner  = raw_input("Did " + players[0].upper() + " win (y/n/-)? ")
    print ""
    print "Summary:"
    print players[0].upper() + " versus " + players[1].upper()
    print "Game duration (min): %.1f" %  gameDuration
    if stripes.lower() == 'y':
        print players[0].upper() + " was stripes."
        print players[1].upper() + " was solids."
        stripesPlayer = players[0].upper()
        solidsPlayer  = players[1].upper()
    elif stripes == 'n':
        print players[0].upper() + " was solids."
        print players[1].upper() + " was stripes."
        stripesPlayer = players[1].upper()
        solidsPlayer  = players[0].upper()
    else:
        print "Invalid entry."
        stripesPlayer = 'NaN' 
        solidsPlayer  = 'NaN' 
    if winner == '-':
        print "No winner."
        winnerPlayer = 'NaN'
    elif winner.lower() == 'y':
        print players[0].upper() + " won."
        print players[1].upper() + " lost."
        winnerPlayer = players[0].upper()
    elif winner.lower() == 'n':
        print players[0].upper() + " lost."
        print players[1].upper() + " won."
        winnerPlayer = players[1].upper()
    else:
        print "Invalid entry."
        winnerPlayer = 'NaN'    
    
    try:
        with open(filename + "_meta.csv", "w") as text_file:
            text_file.write("Player1,Player2,Winner,Stripes,Solids,Duration\n")
            text_file.write("%s,%s,%s,%s,%s,%.3f\n" % (players[0].upper(),players[1].upper(),winnerPlayer,stripesPlayer,solidsPlayer,gameDuration))
    except:
        print "Meta fail."

#    try:
#        proc = Popen('sudo mv -fv *.h264 remoteVideoStorage', shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
#        print "Video files moved to remote storage."
#    except:
#        print "Video files FAILED to be moved to remote storage", (sys.exc_info()[0])
    
#    try:
#        os.system("python stats.py")
#        print "Stats updated."
#    except:
#        print "Failed to update stats."
   
    time.sleep(3)
