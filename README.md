# Social-Distancing using Client Server

Due to the Covid-19 sanitary rules, people have to keep meter apart from each other.
The project aim to calculate distance between people using [Yolo-v3](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/ "Named link title") implementation of Object Detection.  
  My client side convert my video into frames to send them to the server. This one will use our saved model of Object detection to predict on new frame we just received from the client side,then keep all the 'persons' objects box. We will after this calculate Euclidean distance between all centroid box, and finally return the safe persons and too close people.
  
  
 ## Result 
 ![alt text](https://github.com/sodi16/Social_Distancing_Detector/blob/main/result_social_distancing.gif?raw=true)
    
  
