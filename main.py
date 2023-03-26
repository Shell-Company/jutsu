import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import json
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

lastPoseRatio = None
firstRun = True

# args for tolerance and shortcut file
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tolerance",default=0.03, help="tolerance for pose ratio")
parser.add_argument("-c", "--config", default="pose_shortcuts.json",help="path to json file with shortcuts")
args = parser.parse_args()

tolerance = float(args.tolerance)
shortcutFile = args.config




def calculate_pose_ratio(hand_landmarks):
    global lastPoseRatio
    global firstRun
    global tolerance
    global gesturesDir
    # convert landmarks into a hand object 
    # index finger positions 
    index_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    index_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_finger_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
    index_finger_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
    index_finger_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    index_finger_dip_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z
    index_finger_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    index_finger_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    index_finger_pip_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z
    index_finger_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    index_finger_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    index_finger_mcp_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
    # middle finger positions
    middle_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    middle_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_finger_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z
    middle_finger_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
    middle_finger_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    middle_finger_dip_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z
    middle_finger_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
    middle_finger_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    middle_finger_pip_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z
    middle_finger_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    middle_finger_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    middle_finger_mcp_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
    # ring finger positions
    ring_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
    ring_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_finger_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z
    ring_finger_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
    ring_finger_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
    ring_finger_dip_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z
    ring_finger_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
    ring_finger_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    ring_finger_pip_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z
    ring_finger_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
    ring_finger_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    ring_finger_mcp_z = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z
    # pinky finger positions
    pinky_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    pinky_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_finger_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z
    pinky_finger_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x
    pinky_finger_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
    pinky_finger_dip_z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z
    pinky_finger_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
    pinky_finger_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    pinky_finger_pip_z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z
    pinky_finger_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
    pinky_finger_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
    pinky_finger_mcp_z = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z
    # thumb positions
    thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z
    thumb_ip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    thumb_ip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    thumb_ip_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z
    thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
    thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    thumb_mcp_z = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z
      

    index_finger_matrix = np.array([[index_finger_tip_x,index_finger_tip_y,index_finger_tip_z],[index_finger_dip_x,index_finger_dip_y,index_finger_dip_z],[index_finger_pip_x,index_finger_pip_y,index_finger_pip_z],[index_finger_mcp_x,index_finger_mcp_y,index_finger_mcp_z]])
    middle_finger_matrix = np.array([[middle_finger_tip_x,middle_finger_tip_y,middle_finger_tip_z],[middle_finger_dip_x,middle_finger_dip_y,middle_finger_dip_z],[middle_finger_pip_x,middle_finger_pip_y,middle_finger_pip_z],[middle_finger_mcp_x,middle_finger_mcp_y,middle_finger_mcp_z]])
    ring_finger_matrix = np.array([[ring_finger_tip_x,ring_finger_tip_y,ring_finger_tip_z],[ring_finger_dip_x,ring_finger_dip_y,ring_finger_dip_z],[ring_finger_pip_x,ring_finger_pip_y,ring_finger_pip_z],[ring_finger_mcp_x,ring_finger_mcp_y,ring_finger_mcp_z]])
    pinky_finger_matrix = np.array([[pinky_finger_tip_x,pinky_finger_tip_y,pinky_finger_tip_z],[pinky_finger_dip_x,pinky_finger_dip_y,pinky_finger_dip_z],[pinky_finger_pip_x,pinky_finger_pip_y,pinky_finger_pip_z],[pinky_finger_mcp_x,pinky_finger_mcp_y,pinky_finger_mcp_z]])
    thumb_matrix = np.array([[thumb_tip_x,thumb_tip_y,thumb_tip_z],[thumb_ip_x,thumb_ip_y,thumb_ip_z],[thumb_mcp_x,thumb_mcp_y,thumb_mcp_z]])

    index_finger_tip = np.linalg.norm(index_finger_matrix[0]-index_finger_matrix[1])
    index_finger_dip = np.linalg.norm(index_finger_matrix[1]-index_finger_matrix[2])
    index_finger_pip = np.linalg.norm(index_finger_matrix[2]-index_finger_matrix[3])
    index_finger_mcp = np.linalg.norm(index_finger_matrix[3]-index_finger_matrix[0])
    middle_finger_tip = np.linalg.norm(middle_finger_matrix[0]-middle_finger_matrix[1])
    middle_finger_dip = np.linalg.norm(middle_finger_matrix[1]-middle_finger_matrix[2])
    middle_finger_pip = np.linalg.norm(middle_finger_matrix[2]-middle_finger_matrix[3])
    middle_finger_mcp = np.linalg.norm(middle_finger_matrix[3]-middle_finger_matrix[0])
    ring_finger_tip = np.linalg.norm(ring_finger_matrix[0]-ring_finger_matrix[1])
    ring_finger_dip = np.linalg.norm(ring_finger_matrix[1]-ring_finger_matrix[2])
    ring_finger_pip = np.linalg.norm(ring_finger_matrix[2]-ring_finger_matrix[3])
    ring_finger_mcp = np.linalg.norm(ring_finger_matrix[3]-ring_finger_matrix[0])
    pinky_finger_tip = np.linalg.norm(pinky_finger_matrix[0]-pinky_finger_matrix[1])
    pinky_finger_dip = np.linalg.norm(pinky_finger_matrix[1]-pinky_finger_matrix[2])
    pinky_finger_pip = np.linalg.norm(pinky_finger_matrix[2]-pinky_finger_matrix[3])
    pinky_finger_mcp = np.linalg.norm(pinky_finger_matrix[3]-pinky_finger_matrix[0])
    thumb_tip = np.linalg.norm(thumb_matrix[0]-thumb_matrix[1])
    thumb_ip = np.linalg.norm(thumb_matrix[1]-thumb_matrix[2])
    thumb_mcp = np.linalg.norm(thumb_matrix[2]-thumb_matrix[0])

    index_finger_ratio = np.zeros((4,4))
    middle_finger_ratio = np.zeros((4,4))
    ring_finger_ratio = np.zeros((4,4))
    pinky_finger_ratio = np.zeros((4,4))
    thumb_ratio = np.zeros((3,3))
    
#  matrix multiplication for index finger
    index_finger_ratio[0][0] = index_finger_tip
    index_finger_ratio[0][1] = index_finger_dip
    index_finger_ratio[0][2] = index_finger_pip
    index_finger_ratio[0][3] = index_finger_mcp
    index_finger_ratio[1][0] = index_finger_dip
    index_finger_ratio[1][1] = index_finger_pip
    index_finger_ratio[1][2] = index_finger_mcp
    index_finger_ratio[1][3] = index_finger_tip
    index_finger_ratio[2][0] = index_finger_pip
    index_finger_ratio[2][1] = index_finger_mcp
    index_finger_ratio[2][2] = index_finger_tip
    index_finger_ratio[2][3] = index_finger_dip
    index_finger_ratio[3][0] = index_finger_mcp
    index_finger_ratio[3][1] = index_finger_tip
    index_finger_ratio[3][2] = index_finger_dip
    index_finger_ratio[3][3] = index_finger_pip 
#  matrix multiplication for middle finger
    middle_finger_ratio[0][0] = middle_finger_tip
    middle_finger_ratio[0][1] = middle_finger_dip
    middle_finger_ratio[0][2] = middle_finger_pip
    middle_finger_ratio[0][3] = middle_finger_mcp
    middle_finger_ratio[1][0] = middle_finger_dip
    middle_finger_ratio[1][1] = middle_finger_pip
    middle_finger_ratio[1][2] = middle_finger_mcp
    middle_finger_ratio[1][3] = middle_finger_tip
    middle_finger_ratio[2][0] = middle_finger_pip
    middle_finger_ratio[2][1] = middle_finger_mcp
    middle_finger_ratio[2][2] = middle_finger_tip
    middle_finger_ratio[2][3] = middle_finger_dip
    middle_finger_ratio[3][0] = middle_finger_mcp
    middle_finger_ratio[3][1] = middle_finger_tip
    middle_finger_ratio[3][2] = middle_finger_dip
    middle_finger_ratio[3][3] = middle_finger_pip
#  matrix multiplication for ring finger
    ring_finger_ratio[0][0] = ring_finger_tip
    ring_finger_ratio[0][1] = ring_finger_dip
    ring_finger_ratio[0][2] = ring_finger_pip
    ring_finger_ratio[0][3] = ring_finger_mcp
    ring_finger_ratio[1][0] = ring_finger_dip
    ring_finger_ratio[1][1] = ring_finger_pip
    ring_finger_ratio[1][2] = ring_finger_mcp
    ring_finger_ratio[1][3] = ring_finger_tip
    ring_finger_ratio[2][0] = ring_finger_pip
    ring_finger_ratio[2][1] = ring_finger_mcp
    ring_finger_ratio[2][2] = ring_finger_tip
    ring_finger_ratio[2][3] = ring_finger_dip
    ring_finger_ratio[3][0] = ring_finger_mcp
    ring_finger_ratio[3][1] = ring_finger_tip
    ring_finger_ratio[3][2] = ring_finger_dip
    ring_finger_ratio[3][3] = ring_finger_pip
#  matrix multiplication for pinky finger
    pinky_finger_ratio[0][0] = pinky_finger_tip
    pinky_finger_ratio[0][1] = pinky_finger_dip
    pinky_finger_ratio[0][2] = pinky_finger_pip
    pinky_finger_ratio[0][3] = pinky_finger_mcp
    pinky_finger_ratio[1][0] = pinky_finger_dip
    pinky_finger_ratio[1][1] = pinky_finger_pip
    pinky_finger_ratio[1][2] = pinky_finger_mcp
    pinky_finger_ratio[1][3] = pinky_finger_tip
    pinky_finger_ratio[2][0] = pinky_finger_pip
    pinky_finger_ratio[2][1] = pinky_finger_mcp
    pinky_finger_ratio[2][2] = pinky_finger_tip
    pinky_finger_ratio[2][3] = pinky_finger_dip
    pinky_finger_ratio[3][0] = pinky_finger_mcp
    pinky_finger_ratio[3][1] = pinky_finger_tip
    pinky_finger_ratio[3][2] = pinky_finger_dip
    pinky_finger_ratio[3][3] = pinky_finger_pip
#  matrix multiplication for thumb

    thumb_ratio[0][0] = thumb_tip
    thumb_ratio[0][1] = thumb_ip
    thumb_ratio[0][2] = thumb_mcp
    thumb_ratio[1][0] = thumb_ip
    thumb_ratio[1][1] = thumb_mcp
    thumb_ratio[1][2] = thumb_tip
    thumb_ratio[2][0] = thumb_mcp
    thumb_ratio[2][1] = thumb_tip
    thumb_ratio[2][2] = thumb_ip

# clear the screen
    # os.system('cls' if os.name == 'nt' else 'clear')

    if (firstRun == False):
    # compare the current pose to the last pose using numpy to calculate the difference
      index_finger_ratio_diff = np.subtract(index_finger_ratio, lastPoseRatio["index_finger_ratio"])
      middle_finger_ratio_diff = np.subtract(middle_finger_ratio, lastPoseRatio["middle_finger_ratio"])
      ring_finger_ratio_diff = np.subtract(ring_finger_ratio, lastPoseRatio["ring_finger_ratio"])
      pinky_finger_ratio_diff = np.subtract(pinky_finger_ratio, lastPoseRatio["pinky_finger_ratio"])
      thumb_ratio_diff = np.subtract(thumb_ratio, lastPoseRatio["thumb_ratio"])
      # get gestures directory 
      gesturesDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestures/")
      for filename in os.listdir(gesturesDir):
        if filename.endswith(".pkl"):
          with open(os.path.join(gesturesDir, filename), "rb") as f:
            saved_pose = pickle.load(f)
            if (np.allclose(index_finger_ratio, saved_pose["index_finger_ratio"], atol=tolerance) and np.allclose(middle_finger_ratio, saved_pose["middle_finger_ratio"], atol=tolerance) and np.allclose(ring_finger_ratio, saved_pose["ring_finger_ratio"], atol=tolerance) and np.allclose(pinky_finger_ratio, saved_pose["pinky_finger_ratio"], atol=tolerance) and np.allclose(thumb_ratio, saved_pose["thumb_ratio"], atol=tolerance)):
              stripped_filename = filename.replace("pose_", "")
              stripped_filename = stripped_filename.replace(".pkl", "")
              stripped_filename = stripped_filename.replace("_", " ")

              print("Matched saved pose: " + stripped_filename)
              # espeak os command to speak the label of the pose, asynchrnously
              # prevent respeaking the same pose
              lastSpokenPose = ""
              if (stripped_filename != lastSpokenPose):
                lastSpokenPose = stripped_filename
                os.system("espeak -v en-us -s 150 " + stripped_filename+ " &")
                # if shortcut file exists, run it
                shortcut_json= json.load(open(shortcutFile))
                # pose is in dictionary     
                if stripped_filename in shortcut_json:
                  # run the shortcut
                  os.system(shortcut_json[stripped_filename])
            else:
              # print(("No match for saved pose: " + filename))
              pass
    firstRun = False

    lastPoseRatio = {
        "index_finger_ratio": index_finger_ratio,
        "middle_finger_ratio": middle_finger_ratio,
        "ring_finger_ratio": ring_finger_ratio,
        "pinky_finger_ratio": pinky_finger_ratio,
        "thumb_ratio": thumb_ratio
    }

# function to save and record poses to an array for comparison later
def save_pose(poseRatio):
  global gesturesDir
  # save the pose, to be labeled later
  save_pose_index = input("Save pose? (y/n): ")
  #  write pose to a file with a label, replace spaces with underscores
  if (save_pose_index == "y"):
    label = input("Label: ")
    label = label.replace(" ", "_")
    
    with open(os.path.join(gesturesDir, "pose_"+label + ".pkl"), "wb") as f:
        pickle.dump(poseRatio, f)
    print("Saved pose: " + label)
  else:
    print("Pose not saved")

  
def load_pose(filename):
    # load the pose from a file
    with open(filename, "r") as f:
        pose = pickle.load(f)
    return pose
   
              
# webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        calculate_pose_ratio(hand_landmarks)
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    # save on key press of p
    if cv2.waitKey(5) & 0xFF == 112:
      save_pose(lastPoseRatio)

cap.release()
