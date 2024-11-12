# Flask Facial Emotion Recognition with DeepFace

This is a basic Flask-based app with authentication, webcam image processing, video upload and facial detection with emotion recognition with DeepFace.

## Virtual Environment

Navigate to the folder where the virtual environment will be created, then execute the following commands:

```
python3 -m venv <VirtualEnviornmentFolder>
source <VirtualEnviornmentFolder>\Scripts\activate
```

## Install packages

Navigate to the App's folder and install the required packages.

```
cd <WebAppFolderName>\
pip install -r requirements.txt
```

## Set Up Flask 
```
cd <WebAppFolderName>\
cd ..
set FLASK_APP=<WebAppFolderName>
set FLASK_ENV=<VirtualEnviornmentFolder>
```

## Run the App
```
flask run
```
- Access: http://127.0.0.1:5000 with a web browser.
