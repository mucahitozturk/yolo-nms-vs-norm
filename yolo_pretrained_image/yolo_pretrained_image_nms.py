# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 00:50:38 2022

@author: Cookie
"""

#%% 1. Bölüm
#-----------------------------------------------------------------------------------------------------------------------

import cv2  
import numpy as np

img = cv2.imread("D:\YOLO\yolo_pretrained_image\images\image-small.jpg")
#print(img.shape)

img_with = img.shape[1]
img_height = img.shape[0]

#-----------------------------------------------------------------------------------------------------------------------

#%% 2. Bölüm
img_blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
        "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
        "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
        "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
        "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
        "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

colors = ["0,255,255","0,0,255","255,0,255","0,255,0","255,0,0","255,255,0","0,255,255","255,255,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors, (18,1))

#-----------------------------------------------------------------------------------------------------------------------

#%% 3. Bölüm
#-----------------------------------------------------------------------------------------------------------------------

model = cv2.dnn.readNetFromDarknet("D:\YOLO\pretrained_model\yolov3.cfg", "D:\YOLO\pretrained_model\yolov3.weights")

layers = model.getLayerNames()


output_layer = [layers[i-1] for i in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer) 

#-----------------------------------------------------------------------------------------------------------------------

#%% 4. Bölüm
#-----------------------------------------------------------------------------------------------------------------------


########## NON-MAXSIMUM SUPPRESSION - OPERATION 1 ##########

ids_list = [] #predicted idleri tutucak
boxes_list = [] #predicted bounding boxları tutucak
confidences_list = [] #predicted confidencesi tutucak

########## END OF - OPERATION 1 ##########






#-----------------------------------------------------------------------------------------------------------------------

#%% 5. Bölüm
#-----------------------------------------------------------------------------------------------------------------------
for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        if confidence > 0.20:

            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_with, img_height, img_with, img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

            start_x = int(box_center_x - (box_width/2)) 
            start_y = int(box_center_y - (box_height/2)) 


            ########## NON-MAXSIMUM SUPPRESSION - OPERATION 2 ##########
            
            ids_list.append(predicted_id) #predicted idleri tutucak
            confidences_list.append(float(confidence)) #predicted confidencesi tutucak
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)]) #predicted bounding boxları tutucak. box_width ve box_height değerleri int olarak alınmalıdır. 
            
            
            ########## END OF - OPERATION 2 ##########


########## NON-MAXSIMUM SUPPRESSION - OPERATION 3 #########

max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4) #0.5 bounding boxlarımızın confidence değeri, 0.4 ise trashold değeridir. Maksimum bounding boxlarımızın idlerinin listesini tutuyor.
print(max_ids)
for max_id in max_ids: #listenin içindeki değerleri çekiyoruz.
    max_class_id = max_id
    box = boxes_list[max_class_id]

    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]

    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]

########## END OF - OPERATION 3 ##########

    end_x = start_x + box_width 
    end_y = start_y + box_height 

    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color] 

    label = "{}: {:.2f}%".format(label, confidence*100)
    print("predicted object {}".format(label))


    cv2.rectangle(img, (start_x,start_y),(end_x, end_y), box_color, 1)
    cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1) 

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-----------------------------------------------------------------------------------------------------------------------