import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time

scale = 2

# Load Wire Frames
image_1 = cv2.imread('C:/Users/bjperry/Research/Power-Line_Tests/Images/Blast_Test_3/frame_169.png')
image_1 = cv2.resize(image_1, None, fx=scale, fy=scale)
'''
r1 = cv2.selectROI(image_1)
imCrop_1 = image_1[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]
'''
r1 = np.array([146, 178, 205, 45])
imCrop_1 = image_1[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]

# Load Stable Frames
image_stable = cv2.imread('C:/Users/bjperry/Research/Power-Line_Tests/Images/Blast_Test_3/frame_stable_169.png')
image_stable = cv2.resize(image_stable, None, fx=scale, fy=scale)
'''
r_stable = cv2.selectROI(image_stable)
imCrop_stable = image_stable[int(r_stable[1]):int(r_stable[1]+r_stable[3]), int(r_stable[0]):int(r_stable[0]+r_stable[2])]
'''
r_stable = np.array([143, 162, 168, 65])
imCrop_stable = image_stable[int(r_stable[1]):int(r_stable[1]+r_stable[3]), int(r_stable[0]):int(r_stable[0]+r_stable[2])]

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imCrop_1,None)
kp_stable, des_stable = sift.detectAndCompute(imCrop_stable,None)

i = 0
displacement_frame = np.array([0, 0, 0])
displacement_stable = np.array([0, 0, 0])
time_begin = time.time()

while True:
    
    if i % 783 == 0 and i != 0: #820
        i = 0
    num = i + 206               #169
    
    image_2 = cv2.imread('C:/Users/bjperry/Research/Power-Line_Tests/Images/Blast_Test_3/frame_'\
                         + str(num) + '.png')
    image_2 = cv2.resize(image_2, None, fx=scale, fy=scale)
    
    image_2_stable = cv2.imread('C:/Users/bjperry/Research/Power-Line_Tests/Images/Blast_Test_3/frame_stable_'\
                         + str(num) + '.png')
    image_2_stable = cv2.resize(image_2_stable, None, fx=scale, fy=scale)

    kp2, des2 = sift.detectAndCompute(image_2,None)
    kp2_stable, des2_stable = sift.detectAndCompute(image_2_stable,None)
    
    try :
    
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        matches_stable = bf.match(des_stable,des2_stable)
        matches_stable = sorted(matches_stable, key = lambda x:x.distance)
    
        #image_3 = cv2.drawMatches(imCrop_1,kp1,image_2,kp2,matches[:5],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
        # Apply ratio test
        list_kp1 = []
        list_kp2 = []
        list_kp_stable = []
        list_kp2_stable = []
        for m in matches[:5]:
            list_kp1.append(kp1[m.queryIdx].pt)
            list_kp2.append(kp2[m.trainIdx].pt)
          
        cv2.circle(image_2, (int(list_kp2[0][0]), int(list_kp2[0][1])), 8, [0,   0, 255], 3)
        cv2.circle(image_2, (int(list_kp2[1][0]), int(list_kp2[1][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2, (int(list_kp2[2][0]), int(list_kp2[2][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2, (int(list_kp2[3][0]), int(list_kp2[3][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2, (int(list_kp2[4][0]), int(list_kp2[4][1])), 8, [200, 200, 0], 3)
        #cv2.circle(image_2, (int(list_kp2[5][0]), int(list_kp2[5][1])), 8, [200, 200, 0], 3)
        #cv2.circle(image_2, (int(list_kp2[6][0]), int(list_kp2[6][1])), 8, [200, 200, 0], 3)
        
        X = np.average(np.array(list_kp2)[:, 0])
        Y = np.average(np.array(list_kp2)[:, 1])
        
        X_n = np.average(np.array(list_kp1)[:, 0])
        Y_n = np.average(np.array(list_kp1)[:, 1])
        
        cv2.circle(image_2, (int(X), int(Y)), 8, [0,   255, 0], 5)
        
        cv2.imshow('Final Image Point', image_2)
        
        displacement_frame = np.vstack((displacement_frame, \
                                 [(time.time() - time_begin), \
                                  int(X) - int(X_n + r1[0]), \
                                  int(Y) - int(Y_n + r1[1])]))
        
        for m in matches_stable[:5]:
            list_kp_stable.append(kp_stable[m.queryIdx].pt)
            list_kp2_stable.append(kp2_stable[m.trainIdx].pt)
          
        cv2.circle(image_2_stable, (int(list_kp2_stable[0][0]), int(list_kp2_stable[0][1])), 8, [0,   0, 255], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[1][0]), int(list_kp2_stable[1][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[2][0]), int(list_kp2_stable[2][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[3][0]), int(list_kp2_stable[3][1])), 8, [200, 200, 0], 3)
        cv2.circle(image_2_stable, (int(list_kp2_stable[4][0]), int(list_kp2_stable[4][1])), 8, [200, 200, 0], 3)
        #cv2.circle(image_2_stable, (int(list_kp2_stable[5][0]), int(list_kp2_stable[5][1])), 8, [200, 200, 0], 3)
        #cv2.circle(image_2_stable, (int(list_kp2_stable[6][0]), int(list_kp2_stable[6][1])), 8, [200, 200, 0], 3)
            
        X_stable = np.average(np.array(list_kp2_stable)[:, 0])
        Y_stable = np.average(np.array(list_kp2_stable)[:, 1])
        
        X_stable_n = np.average(np.array(list_kp_stable)[:, 0])
        Y_stable_n = np.average(np.array(list_kp_stable)[:, 1])
        
        cv2.circle(image_2_stable, (int(X_stable), int(Y_stable)), 8, [0,   255, 0], 5)
        
        cv2.imshow('Final Image Stable', image_2_stable)
        
        displacement_stable = np.vstack((displacement_stable, \
                                 [(time.time() - time_begin), \
                                  int(X_stable) - int(X_stable_n + r_stable[0]), \
                                  int(Y_stable) - int(Y_stable_n + r_stable[1])]))
    
    except:
        pass
    
    i += 1
    
    #cv2.imshow('Final Image Point', image_3)

    key = cv2.waitKey(1)
    
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        break

# Plot Data
# Initialize Variables for Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(displacement_stable[:,0], \
         displacement_stable[:,2])
ax1.plot(displacement_frame[:,0], \
         displacement_frame[:,2])
ax1.plot(displacement_frame[:,0], \
         displacement_frame[:,2] - displacement_stable[:,2])
ax1.set_ylim([-80, 80])
ax1.set_title('Average Movement within ROI')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Distance from Camera (unit)')

# Determine variables
N = displacement_frame.shape[0] 
Fs = N/(displacement_frame[len(displacement_frame)-1, 0] - displacement_frame[0, 0])
T = (displacement_frame[len(displacement_frame)-1, 0] - displacement_frame[0, 0])
print('Number of Samples: ', N)
print('Sampling Frquency (Hz): ', Fs)

#Compute and Plot FFT  
xf = np.linspace(0, Fs, N)
#yf_1 = fft(displacement[:, 1])
#yf_1 = 2.0/N * np.abs(yf_1[0:int(N/2)])
yf_2 = fft(displacement_frame[:,2] - displacement_stable[:,2])
yf_2 = 2.0/N * np.abs(yf_2[0:int(N/2)])
ax3 = fig.add_subplot(212)
#ax3.plot(xf[1:int(N/2)], yf_1[1:len(xf)])
ax3.plot(xf[1:int(N/2)], yf_2[1:len(xf)])
ax3.set_xlim([0, 10]) # 0 - 4
ax3.set_title('FFT')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Amplitude')
ax3.grid()

#print('X-Frequency: ', xf[np.where(yf_1 == max(yf_1[1:len(yf_1)]))][0])
print('Frequency: ', xf[np.where(yf_2 == max(yf_2[1:len(yf_2)]))][0])

plt.show()
