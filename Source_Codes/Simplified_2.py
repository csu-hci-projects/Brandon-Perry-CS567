import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import matplotlib.pyplot as plt

def distance(depth_image, r) :
    di1 = depth_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    _, di2 = cv2.threshold(di1, 0, 1000, cv2.THRESH_TRUNC)
    di3 = di2.ravel()
    di3.sort()
    while True :
        if di3[0] == 0 :
            di3 = np.delete(di3, 0, None) 
        else :
            break
    '''
    di3 = di2[int(len(di2)/4):int(len(di2)*3/4)]
    '''
    return di3.mean(), di3

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
    json_temp['controls-autoexposure-auto'] = 'False'
    json_temp['controls-laserpower'] = 70
    json_temp['param-depthunits'] = 5000
    json_temp['controls-laserstate'] = 'on'
    json_temp['stream-fps'] = 60
    json_temp['stream-width'] = 848
    json_temp['stream-height'] = 480

    # ...from Camera 1
    config_1.enable_device(str(connected_devices[0]))
    config_1.enable_stream(rs.stream.depth, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.z16, \
                         json_temp['stream-fps'])  #USB 3: 848, 480, 60
    config_1.enable_stream(rs.stream.color, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.bgr8, \
                         json_temp['stream-fps']) #USB 2: 640, 480, 30
    config_1.enable_stream(rs.stream.infrared, 1, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.y8, \
                         json_temp['stream-fps'])
    
   # ...from Camera 2
    config_2.enable_device(str(connected_devices[1]))
    config_2.enable_stream(rs.stream.depth, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.z16, \
                         json_temp['stream-fps'])  #USB 3: 848, 480, 60
    config_2.enable_stream(rs.stream.color, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.bgr8, \
                         json_temp['stream-fps']) #USB 2: 640, 480, 30 
    config_2.enable_stream(rs.stream.infrared, 1, json_temp['stream-width'], \
                         json_temp['stream-height'], rs.format.y8, \
                         json_temp['stream-fps'])

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

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize Variables for FPS Rate
time_start = time.time()
time_begin = time.time()
frameCounter = 0
fps = None

reset = True

try:
    
    while True:
    
        if reset:
            
            # Initialize Variables for Plotting
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            all_1 = np.zeros((1,2))
            all_2 = np.zeros((1,2))
    
            # Wait for a coherent pair of frames: depth and color
            frames_1 = pipeline_1.wait_for_frames()
            color_frame_1 = frames_1.get_color_frame()
            
            # Wait for a coherent pair of frames: depth and color
            frames_2 = pipeline_2.wait_for_frames()
            color_frame_2 = frames_2.get_color_frame()
    
            # Convert images to numpy arrays
            color_image_1 = np.asanyarray(color_frame_1.get_data())
            color_image_2 = np.asanyarray(color_frame_2.get_data())
            
            # Select ROI
            # Manually Select the Region to Measure
            r1 = cv2.selectROI(color_image_1)
            r1 = np.array((r1))
            if r1[2] % 2 == 1 :
                r1[2] = r1[2] - 1
            if r1[3] % 2 == 1 :
                r1[3] = r1[3] - 1
                
            cv2.destroyAllWindows()
                
            r2 = cv2.selectROI(color_image_2)
            r2 = np.array((r2))
            if r2[2] % 2 == 1 :
                r2[2] = r2[2] - 1
            if r2[3] % 2 == 1 :
                r2[3] = r2[3] - 1
                
            cv2.destroyAllWindows()
    
            ROI_Sensor_1 = profile_1.get_device().first_roi_sensor()
            ROI_Sensor_2 = profile_2.get_device().first_roi_sensor()
            roi_1 = ROI_Sensor_1.get_region_of_interest()
            roi_2 = ROI_Sensor_2.get_region_of_interest()
            
            roi_1.max_x = r1[0] + r1[2]
            roi_1.max_y = r1[1] + r1[3]
            roi_1.min_x = r1[0]
            roi_1.min_y = r1[1]
            
            roi_2.max_x = r2[0] + r2[2]
            roi_2.max_y = r2[1] + r2[3]
            roi_2.min_x = r2[0]
            roi_2.min_y = r2[1]
            
            ROI_Sensor_1.set_region_of_interest(roi_1)
            ROI_Sensor_2.set_region_of_interest(roi_2)
            
            reset = False

        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        ir1_frame_1 = frames_1.get_infrared_frame(1)
        
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        ir1_frame_2 = frames_2.get_infrared_frame(1)

        if not depth_frame_1 or not color_frame_1:
            continue

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        ir_image_1 = np.asanyarray(ir1_frame_1.get_data())
        ir_image_2 = np.asanyarray(ir1_frame_2.get_data())
        
        # Get the Average-Middle Distance from the ROI
        depth_1, _ = distance(depth_image_1, r1) * depth_scale_1
        depth_2, di2 = distance(depth_image_2, r2) * depth_scale_2     
        
        # Apply colormap on depth image (image must be converted to 8-bit per 
        # pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1), \
                                           cv2.COLORMAP_JET)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2), \
                                           cv2.COLORMAP_JET)

        # Draw the ROI on the Image
        cv2.rectangle(color_image_1, (r1[0], r1[1]), (r1[0]+r1[2], r1[1]+r1[3]),\
                      color=[255, 255, 0], thickness=2)
        cv2.rectangle(color_image_2, (r2[0], r2[1]), (r2[0]+r2[2], r2[1]+r2[3]),\
                      color=[255, 255, 0], thickness=2)

        # Arrange images

        ir_image_1 = cv2.cvtColor(ir_image_1, cv2.COLOR_GRAY2BGR)
        ir_image_2 = cv2.cvtColor(ir_image_2, cv2.COLOR_GRAY2BGR)
        images_1 = np.hstack((color_image_1, depth_colormap_1, ir_image_1))
        images_2 = np.hstack((color_image_2, depth_colormap_2, ir_image_2))
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
        
        # Save Data to Plot
        all_1 = np.vstack((all_1, [time.time() - time_begin, depth_1]))
        all_2 = np.vstack((all_2, [time.time() - time_begin, depth_2]))

        # Plot Data
        ax.clear()
        ax.plot(all_1[:,0], all_1[:,1])
        ax.plot(all_2[:,0], all_2[:,1])
        ax.set_ylim([0.25, 0.55])
        ax.set_title('Average Movement of the middle values of the ROI')
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance from Camera (meters)')
        plt.show()
        plt.pause(0.00001)
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        # Press r to Reset the ROI
        if key & 0xFF == ord('r'):
            reset = True
            cv2.destroyAllWindows()

finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    