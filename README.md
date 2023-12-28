# Flutter_ObjectDetection

Code originally cloned from : [https://github.com/jcrisp88/flutter-webrtc_python-aiortc-opencv](url)

## Setup Server Environment
- Install python==3.8 (recommend to use conda and create a new environment)
- pip install -r requirements.txt
- Intsall Pytorch : [https://pytorch.org/](url)
- Install Ultralytics : [https://github.com/ultralytics/ultralytics](url)
- The cryptography package might be updated by the above two steps, reinstall to make sure it's cryptography==3.4.7

## Steps to run
- Change IP address in flutter/lib/src/p2pVideo.dart to server's IP and compile app to phone
- Run on the server main.py
- Choose the Detect option within the app and tap start
