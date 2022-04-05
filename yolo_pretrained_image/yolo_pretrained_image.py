# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 00:50:38 2022

@author: Cookie
"""

#%% 1. Bölümümüzde yüklenen kütüphanelerin import edilmesi

import cv2
import numpy as np

img = cv2.imread("D:\YOLO\yolo_pretrained_image\images\image-small.jpg")
#print(img.shape)

img_with = img.shape[1]
img_height = img.shape[0]

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

#%% 3. Bölüm

model = cv2.dnn.readNetFromDarknet("D:\YOLO\pretrained_model\yolov3.cfg", "D:\YOLO\pretrained_model\yolov3.weights")

#detection yapabilmek için modelimizdeki layerları çekebilmemiz gerekiyor.

layers = model.getLayerNames()

#Bu katmanlarda detection işleminin outputlarının bulunduğu katmanlar bulunuyor. Bu aşamada detection yapabilmemiz için tüm katmanlar değil sadece detection katmanlarına ihtiyacımız var.

output_layer = [layers[i-1] for i in model.getUnconnectedOutLayers()] #getUnconnectedOutLayers() fonksiyonu ile detection katmanlarının indexleri bulunuyor.

# output_layer da -1 yapmamızın nedeni indexleri 0 dan başladığı için -1 yapıyoruz.


#modele input olarak resmi verebilmek için blob formata yani 4 boyutlu tensore çeviriyoruz.

model.setInput(img_blob)

detection_layers = model.forward(output_layer) #detection layerları içinde saklayacak olan değişkeni oluşturuyoruz. çıktı katmanlarını model.forward metodu içine sokarak detection layetları elde ediyoruz.


#%% 4. Bölüm

# Daha önceden elde ettiğimiz detection_layersları inceliyoruz. Burada detection layers daki 3 array i teker teker geziyoruz.

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        #Burada tutulan değerlerin ilk 5 i bounding box ile ilgili değerler ondan dolayı buradaki değerleri daha sonra kullanıcaz. 5 den snraki güven skoru değerlerini çekiyoruz.
        
        scores = object_detection[5:] #5 den sonraki değerleri scores değişkenine atıyoruz.


        predicted_id = np.argmax(scores) #en yüksek scores değerlerini max ile buluyoruz.

        confidence = scores[predicted_id] #güven skoru değerini confidence değişkenine atıyoruz.

        if confidence > 0.50:
            #Güven skoru %30 un üzerinde olanlar içeri giriyor. Burada önce labelları çekiyoruz.

            label = labels[predicted_id]

            bounding_box = object_detection[0:4] * np.array([img_with, img_height, img_with, img_height]) #sadece 0:4 arası değerler bounding boxlar için yeterli olmuyo onun için en ve boyları da dahil ediyoruz. bu daha çok yolo nun matematiksel arka planı nedeniyle bu şekilde kullanılıyor.

            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int") #dikdörtgen çizebilmek için box ın merkezi vb. bilgilerine ihtiyacımız var. Burada float değerlerle bounding boxlarda (rectangle) çalışamıyoruz onun için integer a çeviriyoruz.

            start_x = int(box_center_x - (box_width/2)) # x noktası
            start_y = int(box_center_y - (box_height/2)) # y noktası

            end_x = start_x + box_width # x noktasının son noktası
            end_y = start_y + box_height # y noktasının son noktası

            box_color = colors[predicted_id] #boxların rengini belirleyelim.
            box_color = [int(each) for each in box_color] #boxların rengini liste halinde belirleyelim.


            #güven değerini konsolda da görmek için bu 2 satırı kullanıyoruz.
            label = "{}: {:.2f}%".format(label, confidence*100) #label değişkenini formatla.
            print("predicted object {}".format(label))


            cv2.rectangle(img, (start_x,start_y),(end_x, end_y), box_color, 1) #rectangle fonksiyonu ile boxları çizdirelim.

            cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1) #boxların üzerine labelları yazdıralım.

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()