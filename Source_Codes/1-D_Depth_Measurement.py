import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Function to calculate the Average Depth in the ROI
def distance(depth_image, r, depth_scale) :
    
    di1 = depth_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] \
          * depth_scale

    try :
    
        di2 = np.where((di1 > 0.55) | (di1 <= 0.25), 0, di1)
        di3 = di2.ravel()
        di3.sort()
        while True :
            if di3[0] == 0 :
                di3 = np.delete(di3, 0, None) 
            else :
                break
        out =  di3.mean()
        
    except: 
        out = 0
    
    return out

# Define whether to live stream or upload a file 
# 1 - Live Stream
# 2 - Load .bag File
# 3 - Load .txt File
technique = 2

if technique == 1 or technique == 2 :

    # Load files if applicable here:
    if technique == 2 :
        file_name_1 = 'Bag_File_1.bag'
        file_name_2 = 'Bag_File_2.bag'
    
    # Load in Cameras if Live Stream was selected
    if technique == 1 :
    
        # Get Miltiple Cameras
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
        with open("Settings.json", \
                  'r') as file:
            json_temp = json.load(file)
            json_temp['controls-autoexposure-auto'] = 'True'
            json_temp['controls-laserpower'] = 150
            json_temp['param-depthunits'] = 5000
            json_temp['controls-laserstate'] = 'on'
            json_temp['stream-fps'] = 30
            json_temp['stream-width'] = 424 #848
            json_temp['stream-height'] = 240 #480
        
            # ...from Camera 1
            config_1.enable_device(str(connected_devices[0]))
            config_1.enable_stream(rs.stream.depth, json_temp['stream-width'], \
                                 json_temp['stream-height'], rs.format.z16, \
                                 json_temp['stream-fps'])  #USB 3: 848, 480, 60
                                                           #USB 2: 640, 480, 30
            config_1.enable_stream(rs.stream.infrared, 1, json_temp['stream-width'], \
                                 json_temp['stream-height'], rs.format.y8, \
                                 json_temp['stream-fps'])
            
           # ...from Camera 2
            config_2.enable_device(str(connected_devices[1]))
            config_2.enable_stream(rs.stream.depth, json_temp['stream-width'], \
                                 json_temp['stream-height'], rs.format.z16, \
                                 json_temp['stream-fps'])  #USB 3: 848, 480, 60
                                                           #USB 2: 640, 480, 30 
            config_2.enable_stream(rs.stream.infrared, 1, json_temp['stream-width'], \
                                 json_temp['stream-height'], rs.format.y8, \
                                 json_temp['stream-fps'])
        
        # Start streaming
        profile_1 = pipeline_1.start(config_1)
        profile_2 = pipeline_2.start(config_2)
            
        with open("Settings_temp.json", \
                  'w') as file:
            json.dump(json_temp, file, indent=2)
            
        with open("Settings_temp.json", \
                  'r') as file:
            json_text = file.read()
            # Get the active profile and load the json file which contains settings 
            # readable by the realsense
            advanced_mode_1 = rs.rs400_advanced_mode(profile_1.get_device())
            advanced_mode_1.load_json(json_text)
            advanced_mode_2 = rs.rs400_advanced_mode(profile_2.get_device())
            advanced_mode_2.load_json(json_text)
     
    # Load in File if .bag file was selected       
    if technique == 2 :
        # Configure depth and color streams...
        # ...from Camera 1
        pipeline_1 = rs.pipeline()
        config_1 = rs.config()
        
        # ...from Camera 2
        pipeline_2 = rs.pipeline()
        config_2 = rs.config()
        
        config_1.enable_device_from_file(file_name_1)
        config_2.enable_device_from_file(file_name_2)
        
        # Start streaming
        profile_1 = pipeline_1.start(config_1)
        profile_2 = pipeline_2.start(config_2)
    
    # Getting the depth sensor's depth scale
    depth_sensor_1 = profile_1.get_device().first_depth_sensor()
    depth_scale_1 = depth_sensor_1.get_depth_scale()
    depth_sensor_2 = profile_2.get_device().first_depth_sensor()
    depth_scale_2 = depth_sensor_2.get_depth_scale()
    
    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # Initialize some Variables
    reset = True
    displacement = np.array([0, 0, 0])
    frame = 0
    
    try:
        
        while True:
        
            if reset:
        
                 # Wait for a coherent pair of frames: depth and color
                frames_1 = pipeline_1.wait_for_frames()
                ir1_frame_1 = frames_1.get_infrared_frame(1)
    
                
                # Wait for a coherent pair of frames: depth and color
                frames_2 = pipeline_2.wait_for_frames()
                ir1_frame_2 = frames_2.get_infrared_frame(1)
        
                # Convert images to numpy arrays
                ir_image_1 = np.asanyarray(ir1_frame_1.get_data())
                ir_image_2 = np.asanyarray(ir1_frame_2.get_data())
                
                # Select ROI
                # Manually Select the Region to Measure
                r1 = cv2.selectROI(ir_image_1)
                r1 = np.array((r1))
                if r1[2] % 2 == 1 :
                    r1[2] = r1[2] - 1
                if r1[3] % 2 == 1 :
                    r1[3] = r1[3] - 1
                    
                cv2.destroyAllWindows()
                    
                r2 = cv2.selectROI(ir_image_2)
                r2 = np.array((r2))
                if r2[2] % 2 == 1 :
                    r2[2] = r2[2] - 1
                if r2[3] % 2 == 1 :
                    r2[3] = r2[3] - 1
                    
                cv2.destroyAllWindows()
                
                if technique == 1:
                    
                    # If Live Stream, set exposure to ROI
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
                
                # Initialize Variables for FPS Rate
                time_start = time.time()
                time_begin = time.time()
                frameCounter = 0
                fps = None
    
            # Wait for a coherent pair of frames: depth and IR 
            # ...from Camera 1
            frames_1 = pipeline_1.wait_for_frames()
            depth_frame_1 = frames_1.get_depth_frame()
            ir1_frame_1 = frames_1.get_infrared_frame(1)
            
            # Wait for a coherent pair of frames: depth and IR
            # ...from Camera 2
            frames_2 = pipeline_2.wait_for_frames()
            depth_frame_2 = frames_2.get_depth_frame()
            ir1_frame_2 = frames_2.get_infrared_frame(1)
    
            # Convert images to numpy arrays
            depth_image_1 = np.asanyarray(depth_frame_1.get_data())
            depth_image_2 = np.asanyarray(depth_frame_2.get_data())
            ir_image_1 = np.asanyarray(ir1_frame_1.get_data())
            ir_image_2 = np.asanyarray(ir1_frame_2.get_data())
            
            # Get the Average-Middle Distance from the ROI
            depth_1 = distance(depth_image_1, r1, depth_scale_1)
            depth_2 = distance(depth_image_2, r2, depth_scale_2)     
    
            # Draw the ROI on the Image
            cv2.rectangle(ir_image_1, (r1[0], r1[1]), (r1[0]+r1[2], r1[1]+r1[3]),\
                          color=255, thickness=2)
            cv2.rectangle(ir_image_2, (r2[0], r2[1]), (r2[0]+r2[2], r2[1]+r2[3]),\
                          color=255, thickness=2)
    
            # Arrange images
            images = np.vstack((ir_image_1, ir_image_2))
            
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
            
            if technique == 1:
                # Save Data to Plot
                displacement = np.vstack((displacement,  [(time.time() - time_begin), \
                                                          depth_1, depth_2]))
    
            if technique == 2:
                # Save Data to Plot
                frame += 1 
                displacement = np.vstack((displacement,  [(time.time() - time_begin), \
                                                          depth_1, depth_2]))        
    
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

# Only load the Data File with displacements
if technique == 3 :
    file_location = 'Displacement_File.txt'
    t, x, y = np.genfromtxt(file_location, delimiter=' ', unpack=True)
    displacement = np.empty((len(t), 3))
    displacement[:, 0], displacement[:, 1], displacement[:,2] = t[:], x[:], y[:]

# Plot Data
# Initialize Variables for Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(displacement[1:len(displacement),0], \
         displacement[1:len(displacement),1])
ax1.plot(displacement[1:len(displacement),0], \
         displacement[1:len(displacement),2])
ax1.set_ylim([0.25, 0.52]) #[0.25, 1.0])
ax1.set_title('Average Movement within ROI')
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Distance from Camera (meter)')
plt.legend(('Movement in X-Direction', 'Movement in Y-Direction'))

# Determine variables
N = displacement.shape[0] 
Fs = N/(displacement[len(displacement)-1, 0] - displacement[0, 0])
T = (displacement[len(displacement)-1, 0] - displacement[0, 0])

#Compute and Plot FFT  
xf = np.linspace(0, Fs, N)

yf_1 = fft(displacement[:, 1])
yf_1 = 2.0/N * np.abs(yf_1[0:int(N/2)])

yf_2 = fft(displacement[:, 2])
yf_2 = 2.0/N * np.abs(yf_2[0:int(N/2)])

yf_11 = (yf_1 - np.min(yf_1)) / np.ptp(yf_1)
yf_22 = (yf_2 - np.min(yf_2)) / np.ptp(yf_2)

ax3 = fig.add_subplot(212)
ax3.plot(xf[1:int(N/2)], yf_11[1:len(xf)])
ax3.plot(xf[1:int(N/2)], yf_22[1:len(xf)])
ax3.set_xlim([0, 4]) # 0 - 4
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Amplitude')
ax3.grid()

print('X-Frequency: ', xf[np.where(yf_1 == max(yf_1[1:len(yf_1)]))][0])
print('Y-Frequency: ', xf[np.where(yf_2 == max(yf_2[1:len(yf_2)]))][0])

plt.show()
    
