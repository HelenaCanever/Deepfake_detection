import cv2
import math
import numpy as np
import os
from PIL import Image, ImageFilter
import face_recognition
import streamlit as st
import tempfile
import tensorflow as tf
from model_methods import predict

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="👹",
    layout="wide"
)

st.title("Deepfake Detector 👹👨 ")

st.subheader("A deep neural network for the detection of Deepfake videos")

st.write("Upload a video in mp4 format and obtain a prediction. Is your video real or a Deepfake?")

st.caption("Credits:")
st.caption("- developed by: [**Helena Canever**](https://www.linkedin.com/in/helenacanever/), **Dimitry Heu-Mojaïsky**, **Jordan Le Saux** ")
st.caption("- training datasets: [**Facebook Deepfake Detection Challenge dataset**](https://www.kaggle.com/competitions/deepfake-detection-challenge/data), [**Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics**](https://github.com/yuezunli/celeb-deepfakeforensics) (by Yuezun Li and Xin Yang and Pu Sun and Honggang Qi and Siwei Lyu) ")
st.caption("- [source code](https://github.com/HelenaCanever/Deepfake_detection)")


# show example of prediction here

uploaded_video = st.file_uploader("Upload mp4 file: ", type=['mp4'])

t = st.empty()
if uploaded_video is not None:
    my_bar = st.progress(0)
    t.text('Opening video...')
    #get temporary folder 
    tmpdir = tempfile.TemporaryDirectory()
    tmpdirname=tmpdir.name
    os.chdir(tmpdirname)
    #save video in temp folder
    video = uploaded_video.name
    my_bar.progress(30)
    with open(video, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to temp folder
    my_bar.progress(40)
    videocap = cv2.VideoCapture(video) # load video from disk
    frameRate = videocap.get(5)
    t.text('Extracting frames...')
    my_bar.progress(60)
    frames=[]
    while(videocap.isOpened()):
            frameId = videocap.get(1) #current frame number
            ret, frame = videocap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate/5)) == 0 and (frameId <= (frameRate*5)) : 
                filename = video.rstrip('.mp4') +  str(int(frameId)) + ".jpg"
                filepath = os.path.join(tmpdirname, filename)
                frames.append(filename)
                cv2.imwrite(filepath, frame)
    videocap.release()
    my_bar.progress(80)
    n=0
    t.text('Detecting face...')
    images=[]
    for imagename in frames:
        if imagename.endswith(".jpg"):
            path = os.path.join(tmpdirname, imagename)
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)
            #extract coordinates of the first face detected (to avoid pollution from other detections)
            try:
                top, right, bottom, left = face_locations[0]
            except:
                continue

            # calculate 10% padding
            pad = int((bottom-top)*0.1)

            #create image
            try:
                face_image = image[(top-pad):(bottom+pad), (left-pad):(right+pad)]
                pil_image = Image.fromarray(face_image)
            except:
                face_image = image[(top):(bottom), (left):(right)]
                pil_image = Image.fromarray(face_image)
            
            #resize and blur
            im = pil_image.resize((128, 128))
            im = im.filter(ImageFilter.GaussianBlur(radius = 0.5))
            im_path = os.path.join(tmpdirname, str(n) + ".png")
            im = im.save(im_path)
            images.append(str(n) + ".png")
            n+=1
    try:
        frames_path = [os.path.join(tmpdirname,x) for x in images[:10]]
        st.image(frames_path)
        my_bar.progress(100)
        t.text("Face detected!")
    except:
        my_bar.progress(100)
        t.text("No face detected")


    my_bar_2 = st.progress(0)
    x = st.empty()
    x.text('Analyzing...')
    try:
        if len(images)>=10:
            my_bar_2.progress(20)
            Y=[]
            for path in frames_path:
                image = cv2.imread(path,cv2.IMREAD_COLOR)
                Y.append(image)
            my_bar_2.progress(40)
            Y_input_t = tf.convert_to_tensor(np.array(Y))
            Y_input_t = tf.expand_dims(Y_input_t, axis=0)
            os.chdir("..")
            os.chdir("..")
            os.chdir("home/app")

            result, probs = predict(Y_input_t)
            my_bar_2.progress(80)
            st.subheader(f"The video is...{result} with a {round(probs[np.argmax(probs)]*100,2)}% probability")
            x.text('Done!')
            my_bar_2.progress(100)
            tmpdir.cleanup()
    except:
        my_bar_2.progress(100)
        st.subheader('Please upload valid video...')
