import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time



scale = 2.5

image_1 = cv2.imread('C:/Users/bjperry/Research/Power-Line_Tests/Images/Wire/frame_904.jpg')
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_1 = cv2.resize(image_1, None, fx=scale, fy=scale)

'''

r1 = cv2.selectROI(image_1)
imCrop_1 = image_1[int(r1[1]):int(r1[1]+r1[3]), int(qr1[0]):int(r1[0]+r1[2])]

'''

r1 = np.array([207, 77, 50, 173])
imCrop_1 = image_1[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]


# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imCrop_1,None)

i = 0
displacement = np.array([0, 0, 0])
time_begin = time.time()

while True:
    
    if i % 139 == 0:
        i = 0
    num = i + 904
    
    image_2 = cv2.imread('C:/Users/bjperry/Research/Power-Line_Tests/Images/Wire/frame_'\
                         + str(num) + '.jpg')
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.resize(image_2, None, fx=scale, fy=scale)

    
    kp2, des2 = sift.detectAndCompute(image_2,None)
    
    try:
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1[0:1],des2,k=2)
    
        # Apply ratio test
        good = []
        list_kp1 = []
        list_kp2 = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
                list_kp1.append(kp1[m.queryIdx].pt)
                list_kp2.append(kp2[m.trainIdx].pt)
          
        cv2.circle(image_2, (int(list_kp2[0][0]), int(list_kp2[0][1])), 4, 255, 1)
        
        cv2.imshow('Final Image Point', image_2)
        
        displacement = np.vstack((displacement, \
                                 [(time.time() - time_begin), \
                                  int(list_kp2[0][0]) - int(list_kp1[0][0] + r1[0]), \
                                  int(list_kp2[0][1]) - int(list_kp1[0][1] + r1[1])]))
    except:
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1[2:3],des2,k=2)
    
        # Apply ratio test
        good = []
        list_kp1 = []
        list_kp2 = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
                list_kp1.append(kp1[m.queryIdx].pt)
                list_kp2.append(kp2[m.trainIdx].pt)
          
        cv2.circle(image_2, (int(list_kp2[0][0]), int(list_kp2[0][1])), 4, 255, 1)
        
        cv2.imshow('Final Image Point', image_2)
        
        displacement = np.vstack((displacement, \
                                 [(time.time() - time_begin), \
                                  int(list_kp2[0][0]) - int(list_kp1[0][0] + r1[0]), \
                                  int(list_kp2[0][1]) - int(list_kp1[0][1] + r1[1])]))

    i += 1
    
    key = cv2.waitKey(1)
    
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        break


# Plot Data
# Initialize Variables for Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(displacement[1:len(displacement),0], \
         displacement[1:len(displacement),1])
ax1.plot(displacement[1:len(displacement),0], \
         displacement[1:len(displacement),2])
ax1.set_ylim([-40, 80])
ax1.set_title('Average Movement within ROI')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Distance from Camera (unit)')

# Determine variables
N = displacement.shape[0] 
Fs = N/(displacement[len(displacement)-1, 0] - displacement[0, 0])
T = (displacement[len(displacement)-1, 0] - displacement[0, 0])
print('Number of Samples: ', N)
print('Sampling Frquency (Hz): ', Fs)

#Compute and Plot FFT  
xf = np.linspace(0, Fs, N)
yf_1 = fft(displacement[:, 1])
yf_1 = 2.0/N * np.abs(yf_1[0:int(N/2)])
yf_2 = fft(displacement[:, 2])
yf_2 = 2.0/N * np.abs(yf_2[0:int(N/2)])
ax3 = fig.add_subplot(212)
ax3.plot(xf[1:int(N/2)], yf_1[1:len(xf)])
ax3.plot(xf[1:int(N/2)], yf_2[1:len(xf)])
ax3.set_xlim([0, 1]) # 0 - 4
ax3.set_title('FFT')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Amplitude')
ax3.grid()

print('X-Frequency: ', xf[np.where(yf_1 == max(yf_1[1:len(yf_1)]))][0])
print('Y-Frequency: ', xf[np.where(yf_2 == max(yf_2[1:len(yf_2)]))][0])

plt.show()



