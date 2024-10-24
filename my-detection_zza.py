#!/usr/bin/env python3
import jetson.inference
import jetson.utils
from PIL import Image
import numpy as np

# Load the detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Open the camera stream
camera = jetson.utils.videoSource("/dev/video0")  # '/dev/video0' for V4L2

# Create a display output
display = jetson.utils.videoOutput("display://0")  # 'my_video.mp4' for file

# Path to save the captured frame
output_path = '/home/nvidia/jetson-inference/examples/capture.png'

try:
    while display.IsStreaming():
        # Capture the next video frame
        img = camera.Capture()

        if img is None:
            continue  # capture timeout

        # Detect objects in the image
        detections = net.Detect(img)

        # Render the image with detections
        display.Render(img)

        # Update the window title with the current FPS
        display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

        # Save the frame as an image file
        capture_img = jetson.utils.cudaToNumpy(img)
        capture_img = Image.fromarray(capture_img)
        capture_img.save(output_path)

        # Print detection results
        for detection in detections:
            print("ClassID:", detection.ClassID)
            print("Confidence:", detection.Confidence)
            print("Left:", detection.Left)
            print("Top:", detection.Top)
            print("Right:", detection.Right)
            print("Bottom:", detection.Bottom)
            print("Width:", detection.Width)
            print("Height:", detection.Height)
            print("Area:", detection.Area)
            print("Center:", detection.Center)
            print("-" * 40)

        # Break the loop after capturing one frame
        break

except KeyboardInterrupt:
    print("Program terminated by user.")

# Release resources
display.Release()
camera.Release()
