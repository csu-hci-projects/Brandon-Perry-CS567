import numpy as np
import cv2 as cv

# Function to Transform Image to Frequency Domain and then Veiw
def toFreqview(Img) :
    
    rows, cols = Img.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    padded = cv.copyMakeBorder(Img, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexImg = cv.merge(planes)         # Add to the expanded another plane with zeros
    
    cv.dft(complexImg, complexImg, flags=cv.DFT_COMPLEX_OUTPUT) 
    
    cv.split(complexImg, planes)                  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0]) # planes[0] = magnitude
    magImg = planes[0]
    
    matOfOnes = np.ones(magImg.shape, dtype=magImg.dtype)
    cv.add(matOfOnes, magImg, magImg) #  switch to logarithmic scale
    cv.log(magImg, magImg)
    
    magI_rows, magI_cols = magImg.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magImg = magImg[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)
    q0 = magImg[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magImg[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magImg[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magImg[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magImg[0:cx, 0:cy] = q3
    magImg[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magImg[cx:cx + cx, 0:cy] = q2
    magImg[0:cx, cy:cy + cy] = tmp
        
    return magImg

# Function to Quad Swap Iamge to Display
def dftQuadSwap(img) :
    
    img_rows, img_cols = img.shape
    
    cx = int(img_rows / 2)
    cy = int(img_cols / 2)

    if img_rows%2 == 0 and img_cols%2 == 0 :

        q0 = img[0:cx, 0:cy]                # Top-Left - Create a ROI per quadrant
        q1 = img[cx:img_rows, 0:cy]         # Top-Right
        q2 = img[0:cx, cy:img_cols]         # Bottom-Left
        q3 = img[cx:img_rows, cy:img_cols]  # Bottom-Right
        
        tmp = np.copy(q0)                   # swap quadrants (Top-Left with Bottom-Right)
        img[0:cx, 0:cy] = q3
        img[cx:cx + cx, cy:cy + cy] = tmp
        tmp = np.copy(q1)                   # swap quadrant (Top-Right with Bottom-Left)
        img[cx:cx + cx, 0:cy] = q2
        img[0:cx, cy:cy + cy] = tmp
        
    else :        
        if img_rows%2 == 1 :
            img = np.vstack((img, np.zeros((1,img_cols))))
        if img_cols%2 == 1 :
            img = np.hstack((img, np.zeros((img_rows,1))))
            
        img_rows, img_cols = img.shape
    
        cx = int(img_rows / 2)
        cy = int(img_cols / 2)
            
        q0 = img[0:cx, 0:cy]                # Top-Left - Create a ROI per quadrant
        q1 = img[cx:img_rows, 0:cy]         # Top-Right
        q2 = img[0:cx, cy:img_cols]         # Bottom-Left
        q3 = img[cx:img_rows, cy:img_cols]  # Bottom-Right
        
        tmp = np.copy(q0)                   # swap quadrants (Top-Left with Bottom-Right)
        img[0:cx, 0:cy] = q3
        img[cx:cx + cx, cy:cy + cy] = tmp
        tmp = np.copy(q1)                   # swap quadrant (Top-Right with Bottom-Left)
        img[cx:cx + cx, 0:cy] = q2
        img[0:cx, cy:cy + cy] = tmp
    
    return img

# Function to Pass an Averages Sobal Filter over an Image
def sobelAVG(Img) :

    Img = cv.GaussianBlur(Img, (3, 3), 0, 0, cv.BORDER_DEFAULT)

    sobelx = cv.Sobel(Img, cv.CV_32F, 1, 0)
    abs_grad_x = cv.convertScaleAbs(sobelx)

    sobely = cv.Sobel(Img, cv.CV_32F, 0, 1);
    abs_grad_y = cv.convertScaleAbs(sobely)

    # Take the Average of the two directions
    sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    sobel = cv.normalize(sobel, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    return sobel

# Function to Complex Divide two Complex Matrices (Images in the Frequency Domain)
def complexDivide(G, F) :
   
    top = cv.mulSpectrums(G, F, flags=cv.DFT_COMPLEX_OUTPUT, conjB=True) # Top is G*(F conjugate)
    bot = cv.mulSpectrums(F, F, flags=cv.DFT_COMPLEX_OUTPUT, conjB=True) # Bot is F*(F conjugate)
    
    # Bottom is strictly real and we should divide real and complex parts by it
    botRe = [np.float32(bot), np.zeros(bot.shape, np.float32)]
    botRe = cv.split(bot)
    botRe[1] = botRe[0].copy()
    bot = cv.merge(botRe)

    # Do the actual division
    H = np.divide(top, bot)
    
    return H

# Function to Transform Image to Frequency Domain
def toFreq (Img) :
    
    rows, cols = Img.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    padded = cv.copyMakeBorder(Img, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    
    Planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    
    RI = cv.merge(Planes,2) 
    cv.dft(RI, RI, flags=cv.DFT_COMPLEX_OUTPUT)
    
    return RI

#-----Begin Coding-----
    
# Load in Video File
cap = cv.VideoCapture('Video.mp4')

# Change Size of Image for faster Computing
width = int(1280*.6)
length = int(720*.6)

# Initialize some Variables
fri = 0
start = 5
eta = 0.125

# Circle through Video
while(cap.isOpened()):
    
    # Initialize First Frame
    if fri == 0 :
        
        # Load in Video one Frame at a Time
        _, Img = cap.read()
        Img = Img[1500:3000, 1200:2200]
        Img = cv.resize(Img, (width, length), interpolation=cv.INTER_LANCZOS4)
        ImgColor = np.copy(Img)    
        ImgColorDisp = cv.normalize(ImgColor, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)  
        ImgDisp = cv.normalize(Img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)      
        
        # Frame Counter
        fri += 1
        
        # Manually Select the Truck
        r = cv.selectROI(ImgColor, False)
        r = np.array((r))
        if r[2] % 2 == 1 :
            r[2] = r[2] - 1
        if r[3] % 2 == 1 :
            r[3] = r[3] - 1
            
        cv.destroyAllWindows()
        
        # Creates Kernal Gauss Point the size of the Video and places it at
        # the center. Then Converts to Frequency Domain
        cp = np.array([int(r[0]+r[2]/2), int(r[1]+r[3]/2)])
        kernelX = cv.getGaussianKernel(11, 11, cv.CV_32F)
        kernelY = cv.getGaussianKernel(11, 11, cv.CV_32F)
        Gauss = kernelX * np.transpose(kernelY)
        Gauss = cv.normalize(Gauss, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        Gauss = cv.copyMakeBorder(Gauss, cp[1]-5, length-cp[1]-6, cp[0]-6, width-cp[0]-5, cv.BORDER_CONSTANT, 0)
        locationRI = toFreq(Gauss)
        
        # Crop the Truck from the Frame
        imCrop = Img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        # Begin to Track the Truck
        # Perform the Sobel Edge Detection
        sobel = sobelAVG(imCrop)
        sobel = cv.copyMakeBorder(sobel, r[1], length-r[1]-r[3], r[0], width-r[0]-r[2], cv.BORDER_CONSTANT, 0)
        
        # Convert sobel to Frequency Domain
        sobelRI = toFreq(sobel)
    
        # Complex Divide both in Frequency Domain
        filterRI = complexDivide(locationRI, sobelRI)

        # Transform Exact Filter to Spacial Domain
        filt = cv.dft(filterRI, flags=cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT)
        filt = dftQuadSwap(filt)
        filt = cv.normalize(filt, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        pos = (fri-1) % 30
        
        # Initialize exactfilter
        exactfilter = filt
        
        for i in range(start) :
            exactfilter = np.dstack((exactfilter,  np.zeros(filt.shape)))
        
        summ = np.zeros(filt.shape)
     
    # Begin to create more robust filter
    if fri > 0 and fri < start :
        
        # Load in Video one Frame at a Time
        _, Img = cap.read()
        Img = Img[1500:3000, 1200:2200]
        Img = cv.resize(Img, (width, length), interpolation=cv.INTER_LANCZOS4)
        ImgColorDisp = np.copy(Img)
        ImgColorDisp = cv.normalize(ImgColorDisp, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)  
        
        # Frame Counter
        fri += 1
      
        # Begin to Track the Truck
        
        # Crop the Truck from the Frame
        imCrop = Img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        # Perform the Sobel Edge Detection
        sobel = sobelAVG(imCrop)
        sobel = cv.copyMakeBorder(sobel, r[1], length-r[1]-r[3], r[0], width-r[0]-r[2], cv.BORDER_CONSTANT, 0)
        imCrop = cv.copyMakeBorder(imCrop, r[1], length-r[1]-r[3], r[0], width-r[0]-r[2], cv.BORDER_CONSTANT, 0)

        # Convert sobel to Frequency Domain
        sobelRI = toFreq(sobel)
    
        # Complex Divide both in Frequency Domain
        filterRI = complexDivide(locationRI, sobelRI)

        # Transform Exact Filter to Spacial Domain
        filt = cv.dft(filterRI, flags=cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT)
        filt = dftQuadSwap(filt)
        filt = cv.normalize(filt, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        exactfilter[:, :, pos] = filt
        
        # Take the Average of the Exact Filters
        summ = np.zeros(filt.shape)
        for i in range(start) :
            
            summ = summ + exactfilter[:, :, pos]
            
        asefImg = summ/(fri)
        asefImg = cv.normalize(asefImg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
        # Find Max Position
        _, _, _, maxLoc = cv.minMaxLoc(Gauss)
        maxX = maxLoc[0]
        maxY = maxLoc[1]
        cv.rectangle(ImgColorDisp, (maxX - int(r[2]/2), maxY - int(r[3]/2)), \
                     (maxX + int(r[2]/2), maxY + int(r[3]/2)), color=[0, 0, 1],\
                     thickness=2)
        
        # Show Weight of the Truck
        try:
            weight
        except NameError :
            pass
        else : 
            cv.putText(ImgColorDisp, weight, (int(maxX - r[2]/2), int(maxY - r[3]/2 -10)), cv.FONT_HERSHEY_DUPLEX, 0.8, [0, 0, 1])        
        
        # Show all Frames
        asefImgDisp = cv.cvtColor(asefImg, cv.COLOR_GRAY2BGR)
        asefImgDisp = cv.normalize(asefImgDisp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        imCropDisp = cv.cvtColor(imCrop, cv.COLOR_GRAY2BGR)
        imCropDisp = cv.normalize(imCropDisp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        GaussDisp = cv.cvtColor(Gauss, cv.COLOR_GRAY2BGR)
        GaussDisp = cv.normalize(GaussDisp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        DispL = np.concatenate((ImgColorDisp, asefImgDisp), axis=0)
        DispR = np.concatenate((imCropDisp, GaussDisp), axis=0)
        Disp = np.concatenate((DispL, DispR), axis=1)
        cv.imshow("Tracking", Disp)    
     
    # Uses created Filter to Follow Truck
    else :
        
        # Logic to Create Filter
        # Sobel * ASEF->LocationLive
        # LocationLive / Sobel->Filter
        # Filter->ASEF
        
        # Load in Video one Frame at a Time
        _, Img = cap.read()
        Img = Img[1500:3000, 1200:2200]
        Img = cv.resize(Img, (width, length), interpolation=cv.INTER_LANCZOS4)
        ImgColorDisp = np.copy(Img)
        ImgColorDisp = cv.normalize(ImgColorDisp, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        Img = cv.cvtColor(Img, cv.COLOR_BGR2GRAY) 
        
        # Frame Counter
        fri += 1    
        
        # Begin to Track the Truck
        
        # Crop the Truck from the Frame
        imCrop = Img[int(maxY - r[3]/2):int(maxY + r[3]/2), int(maxX - r[2]/2):int(maxX + r[2]/2)]
        
        # Perform the Sobel Edge Detection
        sobel = sobelAVG(imCrop)
        sobel = cv.copyMakeBorder(sobel, int(maxY - r[3]/2), int(length - maxY - r[3]/2), \
                                  int(maxX - r[2]/2), int(width - maxX - r[2]/2), cv.BORDER_CONSTANT, 0)
        imCrop = cv.copyMakeBorder(imCrop, int(maxY - r[3]/2), int(length - maxY - r[3]/2), \
                                  int(maxX - r[2]/2), int(width - maxX - r[2]/2), cv.BORDER_CONSTANT, 0)
        
        # Convert sobel to Frequency Domain
        sobelRI = toFreq(sobel)
        
        # Multiply ASEF by sobel to find Location
        asefRI = toFreq(asefImg)
        locationRI = cv.mulSpectrums(sobelRI, asefRI, flags=cv.DFT_COMPLEX_OUTPUT)
        
        # Transform the location to Spatial Coordinates
        location = cv.dft(locationRI, flags=cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT)
        location = dftQuadSwap(location)
        location = cv.normalize(location, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        location = cv.GaussianBlur(location, (11, 11), 0, 0, cv.BORDER_DEFAULT)
        location = cv.normalize(location, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        # Find Max Position
        _, _, _, maxLoc = cv.minMaxLoc(location)
        maxX = maxLoc[0]
        maxY = maxLoc[1]

        cv.rectangle(ImgColorDisp, (maxX - int(r[2]/2), maxY - int(r[3]/2)), \
                     (maxX + int(r[2]/2), maxY + int(r[3]/2)), color=[0, 0, 1],\
                     thickness=2)
        
        # Create a Gauss Point at the max Location
        # Then converts to Frequency Domain
        kernalXmax = cv.getGaussianKernel(11, 11, cv.CV_32F)
        kernalYmax = cv.getGaussianKernel(11, 11, cv.CV_32F)
        GaussLocation = kernalXmax * np.transpose(kernalYmax)
        GaussLocation = cv.normalize(GaussLocation, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
        if (fri%4) == 1 :
            GaussLocationLive = cv.copyMakeBorder(GaussLocation, maxY - 5, length - maxY - 6, \
                                              maxX - 5, width - maxX - 6, cv.BORDER_CONSTANT, value=0)
        elif (fri%4) == 2 :
            GaussLocationLive = cv.copyMakeBorder(GaussLocation, maxY - 5, length - maxY - 6, \
                                              maxX - 6, width - maxX - 5, cv.BORDER_CONSTANT, value=0)
        elif (fri%4) == 3 :
            GaussLocationLive = cv.copyMakeBorder(GaussLocation, maxY - 6, length - maxY - 5, \
                                              maxX - 5, width - maxX - 6, cv.BORDER_CONSTANT, value=0)
        else :
            GaussLocationLive = cv.copyMakeBorder(GaussLocation, maxY - 6, length - maxY - 5, \
                                              maxX - 6, width - maxX - 5, cv.BORDER_CONSTANT, value=0)

        GaussLocationLive = cv.normalize(GaussLocationLive, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        GaussLocationLiveRI = toFreq(GaussLocationLive)
        
        # Complex Devide both in Frequency Domain
        filterRI = complexDivide(GaussLocationLiveRI, sobelRI)
        
        # Transform Each Filter to Spactial Domain
        filt = cv.dft(filterRI, flags=cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT)
        filt = dftQuadSwap(filt)
        filt = cv.normalize(filt, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        # Find ASEF
        asefImg = (eta * filt) + ((1 - eta) * asefImg)
        
        # Show Weight of the Truck
        try:
            weight
        except NameError :
            pass
        else : 
            cv.putText(ImgColorDisp, weight, (int(maxX - r[2]/2), int(maxY - r[3]/2 -10)), cv.FONT_HERSHEY_DUPLEX, 0.8, [0, 0, 1])
        
        # Show all Frames
        asefImgDisp = cv.cvtColor(asefImg, cv.COLOR_GRAY2BGR)
        asefImgDisp = cv.normalize(asefImgDisp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        imCropDisp = cv.cvtColor(imCrop, cv.COLOR_GRAY2BGR)
        imCropDisp = cv.normalize(imCropDisp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        GaussDisp = cv.cvtColor(location, cv.COLOR_GRAY2BGR)
        GaussDisp = cv.normalize(GaussDisp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        
        DispL = np.concatenate((ImgColorDisp, asefImgDisp), axis=0)
        DispR = np.concatenate((imCropDisp, GaussDisp), axis=0)
        Disp = np.concatenate((DispL, DispR), axis=1)
        cv.imshow("Tracking", Disp)
     
    if cv.waitKey(1) & 0xFF == ord('r') :
        fri=0
    if cv.waitKey(1) & 0xFF == ord('q') :
        break
    if cv.waitKey(1) & 0xFF == ord('w') :
        weight = input("Please input the Truck's Weight: ")

cap.release()
cv.destroyAllWindows()