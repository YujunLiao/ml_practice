import pypylon.pylon as py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


NUM_CAMERAS = 10



# setup demo environment with 10 cameras
os.environ["PYLON_CAMEMU"] = f"{NUM_CAMERAS}"



tlf = py.TlFactory.GetInstance()



# create a device filter for Pylon CamEmu devices
di = py.DeviceInfo()
di.SetDeviceClass("BaslerCamEmu")

# you could also set more device filters like e.g.:
# these are combined as AND 
# di.SetSerialNumber("2134234")



devs = tlf.EnumerateDevices([di, ])




cam_array = py.InstantCameraArray(NUM_CAMERAS)



for idx, cam in enumerate(cam_array):
    cam.Attach(tlf.CreateDevice(devs[idx]))

# %%

cam_array.Open()

# %%

# store a unique number for each camera to identify the incoming images
for idx, cam in enumerate(cam_array):
    camera_serial = cam.DeviceInfo.GetSerialNumber()
    print(f"set context {idx} for camera {camera_serial}")
    cam.SetCameraContext(idx)

# %%

# set the exposure time for each camera
for idx, cam in enumerate(cam_array):
    camera_serial = cam.DeviceInfo.GetSerialNumber()
    print(f"set Exposuretime {idx} for camera {camera_serial}")
    cam.ExposureTimeRaw = 10000

# %%

# wait for all cameras to grab 10 frames
frames_to_grab = 10
# store last framecount in array
frame_counts = [0] * NUM_CAMERAS

# %%



# %%

cam_array.StartGrabbing()
while True:
    with cam_array.RetrieveResult(1000) as res:
        if res.GrabSucceeded():
            img_nr = res.ImageNumber
            cam_id = res.GetCameraContext()
            frame_counts[cam_id] = img_nr
            print(f"cam #{cam_id}  image #{img_nr}")

            # do something with the image ....

            # check if all cameras have reached 100 images
            if min(frame_counts) >= frames_to_grab:
                print(f"all cameras have acquired {frames_to_grab} frames")
                break

cam_array.StopGrabbing()


cam_array.Close()




