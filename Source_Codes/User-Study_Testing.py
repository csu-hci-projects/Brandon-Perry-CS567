import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from collections import deque

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

# Class to Plot in Real-Time (Quickly-ish)
class RealtimePlot:
    def __init__(self, axes, max_entries=100, appen=True):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.axes = axes
        self.appen = appen
        self.max_entries = max_entries
        if appen :
            self.lineplot, = axes.plot([], [], 'b-')
        else :
            self.lineplot, = axes.plot([], [], 'ro')
        self.axes.set_autoscaley_on(True)

    def add(self, x, y):
        if self.appen:
            self.axis_x.append(x)
            self.axis_y.append(y)
            self.lineplot.set_data(self.axis_x, self.axis_y)
            self.axes.relim()
            self.axes.autoscale_view() # rescale the y-axis

        else :
            self.axis_x = x
            self.axis_y = y
            self.lineplot.set_data(self.axis_x, self.axis_y)

    def animate(self, figure, callback, interval = 50):
        import matplotlib.animation as animation
        def wrapper(frame_index):
            self.add(*callback(frame_index))
            if self.appen:
                self.axes.relim(); 
                self.axes.autoscale_view() # rescale the y-axis
            return self.lineplot
        animation.FuncAnimation(figure, wrapper, interval=interval)

# Initialize Some Variables for Plotting
fig, axes = plt.subplots(2, 2)
display1 = RealtimePlot(axes[0, 0])
display1.animate(fig, 0, 0)
display2 = RealtimePlot(axes[0, 1], appen=False)
display2.animate(fig, 0, 0)
display3 = RealtimePlot(axes[1, 0])
display3.animate(fig, 0, 0)
display4 = RealtimePlot(axes[1, 0])
display4.animate(fig, 0, 0)

mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(2500, 250, 700, 750)

plt.show()

display1 = RealtimePlot(axes[0, 0])
display1.axes.set_ylim(0.45, 0.50) #0.45 - 0.50
display1.axes.set_title('Movement')
display2 = RealtimePlot(axes[0, 1], appen=False)
display2.axes.set_ylim(0.45, 0.50)
display2.axes.set_xlim(0.29, 0.33)
display2.axes.set_title('Movement')
display3 = RealtimePlot(axes[1, 0])
display3.axes.set_xlabel('Frequency (Hz)')
display4 = RealtimePlot(axes[1, 1])
display4.axes.set_xlim(0.29, 0.33)
display4.axes.set_xlabel('Movement')

# Initialize some variables for Displaying Results
one = np.ones((1024, 2048))
img1 = one * 255
img2 = one * 200
img3 = one * 0
zeros = np.dstack((one*0, one*0, one*0))
cv2.putText(zeros, "Type 'b' to begin", (50, zeros.shape[0]-50),\
            cv2.FONT_HERSHEY_SIMPLEX, 4, [255, 255, 255])
cv2.namedWindow('Count Down')
cv2.moveWindow('Count Down', 1880, 100)
cv2.imshow('Count Down', zeros)

# Creates window to wait until User is Ready
key = cv2.waitKey(1)  
while key == -1 :
    key = cv2.waitKey(1)   
     
    # Press esc or 'b' to close the image window
    if key & 0xFF == ord('b') or key == 27:
        break

# Load files if applicable here:
test =  4 # 1 2 4 3
final = True # True False

if test == 1 :
    file_name_1 = 'File_1_1.bag'
    file_name_2 = 'File_1_2.bag'

if test == 2 :
    file_name_1 = 'File_2_1.bag'
    file_name_2 = 'File_2_2.bag'
    
if test == 3 :
    file_name_1 = 'File_3_1.bag'
    file_name_2 = 'File_3_2.bag'
    
if test == 4 :
    file_name_1 = 'File_4_1.bag'
    file_name_2 = 'File_4_2.bag'

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
playback_1 = profile_1.get_device().as_playback()
playback_1.set_real_time(False)
profile_2 = pipeline_2.start(config_2)
playback_2 = profile_2.get_device().as_playback()
playback_2.set_real_time(False)

# Getting the depth sensor's depth scale
depth_sensor_1 = profile_1.get_device().first_depth_sensor()
depth_scale_1 = depth_sensor_1.get_depth_scale()
depth_sensor_2 = profile_2.get_device().first_depth_sensor()
depth_scale_2 = depth_sensor_2.get_depth_scale()

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize some Variables
frames_1 = pipeline_1.wait_for_frames()
frames_2 = pipeline_2.wait_for_frames()

displacement = np.array([0, 0, 0])
frame = 0
time_begin = frames_2.get_timestamp()

r1 = np.array([455, 300, 4, 6])
r2 = np.array([455, 300, 4, 6])
    
while True:

    # Wait for a coherent pair of frames: depth and IR 
    # ...from Camera 1
    frames_1 = pipeline_1.wait_for_frames()
    depth_frame_1 = frames_1.get_depth_frame()
    
    # Wait for a coherent pair of frames: depth and IR
    # ...from Camera 2
    frames_2 = pipeline_2.wait_for_frames()
    depth_frame_2 = frames_2.get_depth_frame()

    # Convert images to numpy arrays
    depth_image_1 = np.asanyarray(depth_frame_1.get_data())
    depth_image_2 = np.asanyarray(depth_frame_2.get_data())

    # Get the Average-Middle Distance from the ROI
    depth_1 = distance(depth_image_1, r1, depth_scale_1)
    depth_2 = distance(depth_image_2, r2, depth_scale_2)     

    # Save Data to Plot
    frame += 1 
    displacement = np.vstack((displacement, \
                              [(frames_2.get_timestamp() - time_begin)/500, \
                              depth_1, depth_2]))

    # Determine variables
    if len(displacement) % 30 == 0:
        N = displacement.shape[0] 
        Fs = N/(displacement[len(displacement)-1, 0] - displacement[0, 0])
        
        #Compute and Plot FFT  
        xf = np.linspace(0, Fs, N)
        
        yf_1 = fft(displacement[:, 1])
        yf_1 = 2.0/N * np.abs(yf_1[0:int(N/2)])
        
        yf_2 = fft(displacement[:, 2])
        yf_2 = 2.0/N * np.abs(yf_2[0:int(N/2)])
        
        yf_11 = (yf_1[1:int(N/2)] - np.min(yf_1[1:int(N/2)])) / np.ptp(yf_1[1:int(N/2)])
        yf_22 = (yf_2[1:int(N/2)] - np.min(yf_2[1:int(N/2)])) / np.ptp(yf_2[1:int(N/2)])
        
        ax2 = display3.axes
        ax2.clear()
        ax2.plot(2*xf[1:int(N/2)], yf_11, 'g-')
        ax2.plot(2*xf[1:int(N/2)], yf_22, 'g-')
        ax2.grid()
        ax2.set_xlim([0, 4]) # 0 - 4
        ax2.set_xlabel('Frequency (Hz)')
        
    # Plots the Results
    display2.add(displacement[frame, 2], displacement[frame, 1])
    display1.add(displacement[frame, 0], displacement[frame, 1])
    display4.add(displacement[frame, 2], displacement[frame, 0])
    plt.pause(0.00000001)
    
    # Shows the Bright Background
    zeros = np.dstack((img1, img2, img3))
    cv2.putText(zeros, "Frame "+str(frame)+" of 425", (50, zeros.shape[0]-50),\
                cv2.FONT_HERSHEY_SIMPLEX, 4, [0, 0, 0])
    
    # Show images
    cv2.imshow('Count Down', zeros)
    
    # Allows to exit Study
    key = cv2.waitKey(1)
    if frame == 425 :
        cv2.destroyAllWindows()
        break

# Stop streaming
pipeline_1.stop()
pipeline_2.stop()
plt.close('all')

# Plot Data
# Initialize Variables for Plotting
if final:
    fig = plt.figure(2)
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
    
    yf_11 = (yf_1[1:int(N/2)] - np.min(yf_1[1:int(N/2)])) / np.ptp(yf_1[1:int(N/2)])
    yf_22 = (yf_2[1:int(N/2)] - np.min(yf_2[1:int(N/2)])) / np.ptp(yf_2[1:int(N/2)])
    
    ax2 = fig.add_subplot(212)
    ax2.plot(xf[1:int(N/2)], yf_11)
    ax2.plot(xf[1:int(N/2)], yf_22)
    ax2.set_xlim([0, 4]) # 0 - 4
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.grid()
    
    plt.show()
