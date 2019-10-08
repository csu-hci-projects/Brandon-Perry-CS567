import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import matplotlib.pyplot as plt


# Get Cameras
ctx = rs.context()
connected_devices = []
for d in ctx.devices:
    if d.get_info(rs.camera_info.name).lower() != 'platform camera':
        connected_devices.append(d.get_info(rs.camera_info.serial_number))
        
# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()

# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()

# Load Setting
with open("E:/CS 567 - 3D User Interface/RealSense/JSONs/Settings.json", \
          'r') as file:
    json_temp = json.load(file)
    json_temp['controls-laserpower'] = 100
    json_temp['param-depthunits'] = 5000
    json_temp['controls-laserstate'] = 'on'
    json_temp['stream-fps'] = 30
    json_temp['stream-width'] = 640
    json_temp['stream-height'] = 480

    # ...from Camera 1
    config_1.enable_device(str(connected_devices[0]))
    config_1.enable_stream(rs.stream.depth, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.z16, \
                         json_temp['stream-fps'])  #USB 3: 848, 480, 60
    config_1.enable_stream(rs.stream.color, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.bgr8, \
                         json_temp['stream-fps']) #USB 2: 640, 480, 30
    
   # ...from Camera 2
    config_2.enable_device(str(connected_devices[1]))
    config_2.enable_stream(rs.stream.depth, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.z16, \
                         json_temp['stream-fps'])  #USB 3: 848, 480, 60
    config_2.enable_stream(rs.stream.color, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.bgr8, \
                         json_temp['stream-fps']) #USB 2: 640, 480, 30  

# Start streaming
profile_1 = pipeline_1.start(config_1)
profile_2 = pipeline_2.start(config_2)
    
with open("E:/CS 567 - 3D User Interface/RealSense/JSONs/Settings_temp.json", \
          'w') as file:
    json.dump(json_temp, file, indent=2)
    
with open("E:/CS 567 - 3D User Interface/RealSense/JSONs/Settings_temp.json", \
          'r') as file:
    json_text = file.read()
    # Get the active profile and load the json file which contains settings 
    # readable by the realsense
    advanced_mode_1 = rs.rs400_advanced_mode(profile_1.get_device())
    advanced_mode_1.load_json(json_text)
    advanced_mode_2 = rs.rs400_advanced_mode(profile_2.get_device())
    advanced_mode_2.load_json(json_text)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor_1 = profile_1.get_device().first_depth_sensor()
depth_scale_1 = depth_sensor_1.get_depth_scale()
depth_sensor_2 = profile_2.get_device().first_depth_sensor()
depth_scale_2 = depth_sensor_2.get_depth_scale()
print("Sensor 1 Depth Scale is: {0}".format(round(depth_scale_1, 5)))
print("Sensor 2 Depth Scale is: {0}".format(round(depth_scale_2, 5)))

# Initialize Variables for FPS Rate
time_start = time.time()
time_begin = time.time()
frameCounter = 0
fps = None

# Initialize Variables for Plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
all_1 = np.zeros((1,2))
all_2 = np.zeros((1,2))

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        
        # Get Distance from Center Point
        width = depth_frame_1.get_width()
        height = depth_frame_1.get_height()
        center_1 = depth_frame_1.get_distance(int(width/2), int(height/2))
        center_2 = depth_frame_2.get_distance(int(width/2), int(height/2))
        
        if not depth_frame_1 or not color_frame_1:
            continue

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per 
        # pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1), \
                                           cv2.COLORMAP_JET)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2), \
                                           cv2.COLORMAP_JET)

        # Arrange images
        cv2.rectangle(color_image_1, (int(width/2)-2, int(height/2)-2), \
                     (int(width/2)+2, int(height/2)+2), color=[0, 0, 255],\
                     thickness=5)
        cv2.rectangle(color_image_2, (int(width/2)-2, int(height/2)-2), \
                     (int(width/2)+2, int(height/2)+2), color=[0, 0, 255],\
                     thickness=5)
        images_1 = np.hstack((color_image_1, depth_colormap_1))
        images_2 = np.hstack((color_image_2, depth_colormap_2))
        images = np.vstack((images_1, images_2))
        
        # Calculate Frame Rate
        frameCounter += 1
        if time.time() - time_start > 1 :
            fps = frameCounter
            time_start = time.time()
            frameCounter = 0 
        cv2.putText(images, "Frame Rate: "+str(fps), (50, images.shape[0]-50),\
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, [255, 200, 0])
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # Save Data to Plot
        all_1 = np.vstack((all_1, [time.time() - time_begin, center_1]))
        all_2 = np.vstack((all_2, [time.time() - time_begin, center_2]))

        # Plot Data
        ax.clear()
        ax.plot(all_1[:,0], all_1[:,1])
        ax.plot(all_2[:,0], all_2[:,1])
        ax.set_ylim([0.2, 0.5])
        ax.set_title('Movement at Center')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance from Camera (meters)')
        plt.show()
        plt.pause(0.00001)

finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
