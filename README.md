# Jutsu 

```
     ▄█ ███    █▄      ███        ▄████████ ███    █▄  
    ███ ███    ███ ▀█████████▄   ███    ███ ███    ███ 
    ███ ███    ███    ▀███▀▀██   ███    █▀  ███    ███ 
    ███ ███    ███     ███   ▀   ███        ███    ███ 
    ███ ███    ███     ███     ▀███████████ ███    ███ 
    ███ ███    ███     ███              ███ ███    ███ 
    ███ ███    ███     ███        ▄█    ███ ███    ███ 
█▄ ▄███ ████████▀     ▄████▀    ▄████████▀  ████████▀  
▀▀▀▀▀▀                                                 
```

This tool allows you to record hand gestures and then use them to control your computer. Gestures can be saved by pressing the 'p' key while the media window is in focus; saved gestures are immediately available for use. Gestures can be deleted by removing the corresponding file from the gestures folder. 

Tested on Ubuntu 20.04 and MacOS 13.0
# Why Jutsu?
Jutsu is a tool that allows you to control your computer using hand gestures. We hope this tool will help people with disabilities to use their computers more easily. It was inspired by the hand gestures used by ninjas.

# Security Considerations
This tool will work when the screen is locked. This means that anyone who has access to your computer can use the gestures to control your computer. We recommend that you use this tool only when you are the only person who has access to your computer. Additionally you may want to avoid common gestures for sensitive or destruction actions. 

Alternatively, you can use this tool to control your computer when you are logged in but not actively using it. This allows for some interesting use offensive security use cases if you are able to install the tool on a target machine.

# Installation

1. Clone the repository
```
git clone shell-company/jutsu
```
2. Install pre-requisites
```
pip install poetry
brew install espeak || sudo apt install espeak
```

2. Install the requirements
```
poetry install
```

3. Run the main.py file
```
poetry run python main.py
```

4. Press 'p' to save a gesture - with focus on the media player window
5. update the pose_shortcuts.json file to map gestures to actions

# Usage
```
usage: main.py [-h] [-t TOLERANCE] [-c CONFIG]

options:
  -h, --help            show this help message and exit
  -t TOLERANCE, --tolerance TOLERANCE
                        tolerance for pose ratio
  -c CONFIG, --config CONFIG
                        path to json file with shortcuts
```
# Shortcut Configuration File

```
{
    "black-power": "osascript -e 'set volume output muted not (output muted of (get volume settings))'",
    "gunfinger": "open https://www.youtube.com/watch?v=qzCHc9faIUM",
    "peace": "open https://www.reddit.com",
    "noirgate": "open https:/shellz.wtf",
    "terminal": "osascript -e 'tell application \"iTerm\" to activate'",
    "spotlight-search":" osascript -e 'tell application \"System Events\" to key code 49 using command down '" ,
    "close-application": "osascript -e 'tell application \"System Events\" to key code 13 using command down'",
    "up":"osascript -e 'set volume output volume (output volume of (get volume settings) + 5) --100%'",
    "down":"osascript -e 'set volume output volume (output volume of (get volume settings) - 5) --100%'"
  }    

  ```

# Calibrating The Tool
The tool uses mediapipe to detect hand gestures. Mediapipe uses a pre-trained model to detect hand position. For saved poses use the tolerance value to adjust the model to the most accurate value for your system. The default of 0.03 works well for most systems.

# Pre-Requisites
jutsu requires the following to be installed on your system:
- espeak
- mediapipe
- opencv

# Limitations
- The tool currently only supports single gesture shortcuts
- The tool currently only supports gestures that are performed with one hand
- Gestures that are similar to each other may be confused for each other


# Todo
- [ ] Add support for multi-gesture shortcuts
- [ ] Add support for gesture chaining
- [ ] Add support for gesture recording without keyboard input
 