import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

#Initialize Scale Factors
scale = 2 # To Enlarge Image and find better SIFT Points
img_scale = 745.716 # 745.716 - pixel/meter

# Load in the Video
capture = cv2.VideoCapture('Videos.mp4')
capture.set(cv2.CAP_PROP_POS_FRAMES, 169)
_, frame = capture.read()

# Crop Video Exactly at ROI
r1 = np.array([2040, 1064, 105, 17])
r_stable = np.array([1579, 1116, 143, 32])

imCrop_1 = frame[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]
imCrop_1 = cv2.resize(imCrop_1, None, fx=scale, fy=scale)

imCrop_stable = frame[int(r_stable[1]):int(r_stable[1]+r_stable[3]), int(r_stable[0]):int(r_stable[0]+r_stable[2])]
imCrop_stable = cv2.resize(imCrop_stable, None, fx=scale, fy=scale)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors with SIFT for ROI and Background
kp1, des1 = sift.detectAndCompute(imCrop_1,None)
kp_stable, des_stable = sift.detectAndCompute(imCrop_stable,None)

# Initialize arrays
displacement_frame = np.array([0, 0, 0])
displacement_stable = np.array([0, 0, 0])
time_begin = capture.get(cv2.CAP_PROP_POS_MSEC)

# Circle Through Entire Video
while True:
    
    _, frame = capture.read()

    # Crop each frame, but add 100 pixel border in all directions
    image_2 = frame[int(r1[1]-100):int(r1[1]+r1[3]+100), int(r1[0]-100):int(r1[0]+r1[2]+100)]
    image_2 = cv2.resize(image_2, None, fx=scale, fy=scale)
    
    image_2_stable = frame[int(r_stable[1]-100):int(r_stable[1]+r_stable[3]+100), int(r_stable[0]-100):int(r_stable[0]+r_stable[2]+100)]
    image_2_stable = cv2.resize(image_2_stable, None, fx=scale, fy=scale)

    # Compute Keyppoints and Descriptors of newly loaded frame
    kp2, des2 = sift.detectAndCompute(image_2,None)
    kp2_stable, des2_stable = sift.detectAndCompute(image_2_stable,None)
    
    try :
    
        # Match Newly Loaded Frames with the Initial Position Frames
        # And sort to identify best matches using Brute Force
        bf = cv2.BFMatcher()
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        matches_stable = bf.match(des_stable,des2_stable)
        matches_stable = sorted(matches_stable, key = lambda x:x.distance)
        
        # Initialize some arrays
        list_kp1 = []
        list_kp2 = []
        list_kp_stable = []
        list_kp2_stable = []
        
        # Select top 5 Matches
        for m in matches[:5]:
            list_kp1.append(kp1[m.queryIdx].pt)
            list_kp2.append(kp2[m.trainIdx].pt)
            
        for m in matches_stable[:5]:
            list_kp_stable.append(kp_stable[m.queryIdx].pt)
            list_kp2_stable.append(kp2_stable[m.trainIdx].pt)
          
        # Average the Pixel Locations of the Keypoints 
        X = np.average(np.array(list_kp2)[:, 0])
        Y = np.average(np.array(list_kp2)[:, 1])
        
        X_n = np.average(np.array(list_kp1)[:, 0])
        Y_n = np.average(np.array(list_kp1)[:, 1])
        
        X_stable = np.average(np.array(list_kp2_stable)[:, 0])
        Y_stable = np.average(np.array(list_kp2_stable)[:, 1])
        
        X_stable_n = np.average(np.array(list_kp_stable)[:, 0])
        Y_stable_n = np.average(np.array(list_kp_stable)[:, 1])
        
        # Draw points on Image for Visulaization
        cv2.circle(image_2, (int(list_kp2[0][0]), int(list_kp2[0][1])), 8, [0,   0, 255], 3)
        cv2.circle(image_2, (int(list_kp2[1][0]), int(list_kp2[1][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2, (int(list_kp2[2][0]), int(list_kp2[2][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2, (int(list_kp2[3][0]), int(list_kp2[3][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2, (int(list_kp2[4][0]), int(list_kp2[4][1])), 8, [200, 200, 0], 3)

        cv2.circle(image_2, (int(X), int(Y)), 8, [0, 255, 0], 5)
        
        cv2.circle(image_2_stable, (int(list_kp2_stable[0][0]), int(list_kp2_stable[0][1])), 8, [0,   0, 255], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[1][0]), int(list_kp2_stable[1][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[2][0]), int(list_kp2_stable[2][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[3][0]), int(list_kp2_stable[3][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[4][0]), int(list_kp2_stable[4][1])), 8, [200, 200, 0], 3)

        cv2.circle(image_2_stable, (int(X_stable), int(Y_stable)), 8, [0, 255, 0], 5)

        # Save Time and Displacement Information for later processing
        displacement_frame = np.vstack((displacement_frame, \
                                 [(capture.get(cv2.CAP_PROP_POS_MSEC) - time_begin)/1000, \
                                  (int(X) - int(X_n) - 100)/img_scale, \
                                  (int(Y) - int(Y_n) - 100)/img_scale]))
    
        displacement_stable = np.vstack((displacement_stable, \
                                 [(capture.get(cv2.CAP_PROP_POS_MSEC) - time_begin)/1000, \
                                  (int(X_stable) - int(X_stable_n) - 100)/img_scale, \
                                  (int(Y_stable) - int(Y_stable_n) - 100)/img_scale]))

        # Show Image
        cv2.imshow('Final Image Point', image_2)
        cv2.imshow('Final Image Stable', image_2_stable)

    except:
        pass
    
    # Press esc or 'q' to close the image window
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

# Plot Data
# Initialize Variables for Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)

displacement_stable = np.delete(displacement_stable, 36, 0)
displacement_frame = np.delete(displacement_frame, 36, 0)

ax1.plot(displacement_stable[:,0], \
         displacement_stable[:,2])
ax1.plot(displacement_frame[:,0], \
         displacement_frame[:,2])
ax1.plot(displacement_frame[:,0], \
         displacement_frame[:,2] - displacement_stable[:,2])
ax1.set_ylim([-0.050, 0.25])
ax1.set_xlim([1, 17.5])
ax1.set_title('Average Movement within ROI')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Distance from Camera (meter)')
plt.legend(('Movement of UAV', 'Movement of ROI', 'True Movement of ROI'),
           loc='upper right')

# Determine variables
N = displacement_frame.shape[0] 
Fs = N/(displacement_frame[len(displacement_frame)-1, 0] - displacement_frame[0, 0])
T = (displacement_frame[len(displacement_frame)-1, 0] - displacement_frame[0, 0])
print('Results ')
print('------------------')
print('Number of Samples: ', N)
print('Sampling Frquency: % 5.2f'% Fs)

#Compute and Plot FFT  
xf = np.linspace(0, Fs, N)
yf_2 = fft(displacement_frame[:,2] - displacement_stable[:,2])
yf_2 = 2.0/N * np.abs(yf_2[0:int(N/2)])
yf_2 = (yf_2-np.min(yf_2))/np.ptp(yf_2)
ax3 = fig.add_subplot(212)
ax3.plot(xf[1:int(N/2)], yf_2[1:len(xf)])
ax3.set_xlim([0, 10]) # 0 - 4
#ax3.set_title('FFT')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Amplitude')
ax3.grid()

print('        Frequency: % 5.3f'% xf[np.where(yf_2 == max(yf_2[1:len(yf_2)]))][0])

plt.show()

