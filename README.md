## **A Methodology for the Simultaneous Monitoring of Bridge Load and Induced 3-D Displacement using Unmanned Aerial Vehicles (UAVs) with a Real-Time Data Interpretation User Study**

**Overview of Project**

Fully understanding bridge performance during traffic is critical for effective condition assessment. 
Modern structural health monitoring (SHM) has enabled measurements of traffic loads and dynamic responses; 
however, challenges still exist for accurately synchronizing the two inputs. 
Recently, with advancements in unmanned aerial vehicles (UAVs) technology, 
UAVs can hover at specified heights and key locations to provide high-quality sensory data. 
By leveraging UAVs and image computation, this study proposes a UAV-based SHM framework to track vehicular loading and measure the 3-D displacement of 
a bridge simultaneously.
 In the proposed methodology, multiple UAVs will hover adjacent to a bridge to collect data of the moving traffic and structural response. 
Through a case study, the potential of synchronizing dynamic traffic loads and 3-D structural response is demonstrated. 
Lastly, a user study is presented to determine the importance of real-time information compared to post-process information for civil engineers in the field. 

The entire project framework is presented within four modules:

  1. [**Truck Tracking Module:**](https://cs.colostate.edu/~bjperry/index.html#/module-1 "Module 1") In this module, an algorithm is developed to follow and track a large truck (i.e. 18-wheeler) with a UAV from a weigh-in-motion station on the United State's Interstate's Systems to the bridge of interest. The weigh-in-motion stations on the interstate systems will provide the real weight of the trucks. Knowing this weight and having the ability to track its position overtime to the test bridge site will provide the dynamic live loading of a bridge and help develop the first half of an input-output model.
  2. [**2-D Planar Measurement Module:**](https://cs.colostate.edu/~bjperry/index.html#/module-2 "Module 2") Using an optical RGB sensor attached to a UAV hovering normal to a plane of interest, an algorithm is developed which detects and tracks key-points in the image to measure the movement of an object. Since a UAV is hovering during the data recording, a background compensation technique is developed to account for and correct the movement of the UAV to find the true movement of the region of interest. 
  3. [**1-D Depth Measurement Module:**](https://cs.colostate.edu/~bjperry/index.html#/module-3 "Module 3") An Intel RealSense D35 Sensor is used to measure the depth from an object to the sensor. Using a virtual, projected speckle pattern, two infrared cameras are used to calculate depth. This depth measurement or, more importantly, the change of depth measurement provides the third dimensional displacement of an object.
  4. [**User-Study Module:**](https://cs.colostate.edu/~bjperry/index.html#/module-4 "Module 4") Providing the information from this framework in real-time to engineers in the field can be challenging and may require additional hardware and software; therefore, a user study is performed to measure the effectiveness of providing real-time as opposed to post-processed information interpreted in-office. 

The Python Source Code Files can be found in the Source_Code Folder.
The Check Point Notes is found in the Check Point File.
And the Latex Papers can be found in the Latex Folder.

| ![UAV Flight](Brandon-Perry-CS567/Latex_Files/Figures/IMG_2576.jpg "UAV Surveying a Bridge") |
|:--:|
| **UAV Bridge Survey** |
