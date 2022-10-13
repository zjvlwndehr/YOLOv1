import numpy as np
import cv2
import xmltodict
from tqdm import tqdm
import tensorflow as tf
from glob import glob
from tensorflow.keras.callbacks import ModelCheckpoint

# 파일 경로
train_x_path = './data/train/images/*.jpg'
train_y_path = './data/train/labels/*.xml'

test_x_path = './data/test/images/*.jpg'
test_y_path = './data/test/labels/*.xml'

# 파일 경로 휙득
image_file_path_list = sorted([x for x in glob(train_x_path + '/**')])
xml_file_path_list = sorted([x for x in glob(train_y_path + '/**')])

test_image_file_path_list = sorted([x for x in glob(test_x_path + '/**')])
test_xml_file_path_list = sorted([x for x in glob(test_y_path + '/**')])


################################################
################################################
################### Backbone ###################
################################################
################################################




def get_Classes_inImage(xml_file_list):
    Classes_inDataSet = []

    for xml_file_path in xml_file_list: 

        f = open(xml_file_path)
        xml_file = xmltodict.parse(f.read())
        # 사진에 객체가 여러개 있을 경우
        try: 
            for obj in xml_file['annotation']['object']:
                Classes_inDataSet.append(obj['name'].lower()) # 들어있는 객체 종류를 알아낸다
        # 사진에 객체가 하나만 있을 경우
        except TypeError as e: 
            Classes_inDataSet.append(xml_file['annotation']['object']['name'].lower()) 
        f.close()

    Classes_inDataSet = list(set(Classes_inDataSet))
    Classes_inDataSet.sort() # 알파벳 순으로 정렬

    return Classes_inDataSet

def get_label_fromImage(xml_file_path, Classes_inDataSet):

    f = open(xml_file_path)
    xml_file = xmltodict.parse(f.read()) 

    Image_Height = float(xml_file['annotation']['size']['height'])
    Image_Width  = float(xml_file['annotation']['size']['width'])

    label = np.zeros((7, 7, 25), dtype = float)
    
    try:
        for obj in xml_file['annotation']['object']:
            
            class_index = Classes_inDataSet.index(obj['name'].lower())
            
            # min, max좌표 얻기
            x_min = float(obj['bndbox']['xmin']) 
            y_min = float(obj['bndbox']['ymin'])
            x_max = float(obj['bndbox']['xmax']) 
            y_max = float(obj['bndbox']['ymax'])

            # 224*224에 맞게 변형시켜줌
            x_min = float((224.0/Image_Width)*x_min)
            y_min = float((224.0/Image_Height)*y_min)
            x_max = float((224.0/Image_Width)*x_max)
            y_max = float((224.0/Image_Height)*y_max)

            # 변형시킨걸 x,y,w,h로 만들기 
            x = (x_min + x_max)/2.0
            y = (y_min + y_max)/2.0
            w = x_max - x_min
            h = y_max - y_min

            # x,y가 속한 cell알아내기
            x_cell = int(x/32) # 0~6
            y_cell = int(y/32) # 0~6
            # cell의 중심 좌표는 (0.5, 0.5)다
            x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
            y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

            # w, h 를 0~1 사이의 값으로 만들기
            w = w / 224.0
            h = h / 224.0

            class_index_inCell = class_index + 5

            label[y_cell][x_cell][0] = x_val_inCell
            label[y_cell][x_cell][1] = y_val_inCell
            label[y_cell][x_cell][2] = w
            label[y_cell][x_cell][3] = h
            label[y_cell][x_cell][4] = 1.0
            label[y_cell][x_cell][class_index_inCell] = 1.0


    # single-object in image
    except TypeError as e : 
        # class의 index 휙득
        class_index = Classes_inDataSet.index(xml_file['annotation']['object']['name'].lower())
            
        # min, max좌표 얻기
        x_min = float(xml_file['annotation']['object']['bndbox']['xmin']) 
        y_min = float(xml_file['annotation']['object']['bndbox']['ymin'])
        x_max = float(xml_file['annotation']['object']['bndbox']['xmax']) 
        y_max = float(xml_file['annotation']['object']['bndbox']['ymax'])

        # 224*224에 맞게 변형시켜줌
        x_min = float((224.0/Image_Width)*x_min)
        y_min = float((224.0/Image_Height)*y_min)
        x_max = float((224.0/Image_Width)*x_max)
        y_max = float((224.0/Image_Height)*y_max)

        # 변형시킨걸 x,y,w,h로 만들기 
        x = (x_min + x_max)/2.0
        y = (y_min + y_max)/2.0
        w = x_max - x_min
        h = y_max - y_min

        # x,y가 속한 cell알아내기
        x_cell = int(x/32) # 0~6
        y_cell = int(y/32) # 0~6
        x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
        y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

        # w, h 를 0~1 사이의 값으로 만들기
        w = w / 224.0
        h = h / 224.0

        class_index_inCell = class_index + 5

        label[y_cell][x_cell][0] = x_val_inCell
        label[y_cell][x_cell][1] = y_val_inCell
        label[y_cell][x_cell][2] = w
        label[y_cell][x_cell][3] = h
        label[y_cell][x_cell][4] = 1.0
        label[y_cell][x_cell][class_index_inCell] = 1.0

    return label # np array로 반환

def make_dataset(image_file_path_list, xml_file_path_list, Classes_inDataSet) :

    image_dataset = []
    label_dataset = []

    for i in tqdm(range(0, len(image_file_path_list)), desc = "make dataset"):
        image = cv2.imread(image_file_path_list[i]) 
        image = cv2.resize(image, (224, 224))/ 255.0 # 이미지를 넘파이 배열로 불러온 뒤 255로 나눠 픽셀별 R, G, B를 0~1사이의 값으로 만들어버린다.
        label = get_label_fromImage(xml_file_path_list[i], Classes_inDataSet)
        
        image_dataset.append(image)
        label_dataset.append(label)
    
    image_dataset = np.array(image_dataset, dtype="object")
    label_dataset = np.array(label_dataset, dtype="object")
    
    image_dataset = np.reshape(image_dataset, (-1, 224, 224, 3)).astype(np.float32)
    label_dataset = np.reshape(label_dataset, (-1, 7, 7, 25))

    return image_dataset, tf.convert_to_tensor(label_dataset, dtype=tf.float32)


max_num = len(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers) # 레이어 최대 개수

YOLO = tf.keras.models.Sequential(name = "YOLO")
for i in range(0, max_num-1):
    YOLO.add(tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).layers[i])

print(max_num)
YOLO.summary()
# tf.keras.applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224, 224, 3)).summary()

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)  
regularizer = tf.keras.regularizers.l2(0.0005) # L2 규제 == weight decay.

for layer in YOLO.layers:
    # 훈련 X
    layer.trainable=False
    if (hasattr(layer,'activation'))==True:
        layer.activation = leaky_relu


################################################
################################################
##################### Head #####################
################################################
################################################



YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=leaky_relu, kernel_initializer=initializer, kernel_regularizer = regularizer, padding = 'SAME', name = "detection_conv1", dtype='float32'))
YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=leaky_relu, kernel_initializer=initializer, kernel_regularizer = regularizer, padding = 'SAME', name = "detection_conv2", dtype='float32'))
YOLO.add(tf.keras.layers.MaxPool2D((2, 2)))
YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=leaky_relu, kernel_initializer=initializer, kernel_regularizer = regularizer, padding = 'SAME', name = "detection_conv3", dtype='float32'))
YOLO.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=leaky_relu, kernel_initializer=initializer, kernel_regularizer = regularizer, padding = 'SAME', name = "detection_conv4", dtype='float32'))
# Linear 부분
YOLO.add(tf.keras.layers.Flatten())
YOLO.add(tf.keras.layers.Dense(4096, activation=leaky_relu, kernel_initializer = initializer, kernel_regularizer = regularizer, name = "detection_linear1", dtype='float32'))
YOLO.add(tf.keras.layers.Dropout(.5))
# 마지막 레이어의 활성화 함수는 선형 활성화 함수인데 이건 입력값을 그대로 내보내는거라 activation을 따로 지정하지 않았다.
YOLO.add(tf.keras.layers.Dense(1470, kernel_initializer = initializer, kernel_regularizer = regularizer, name = "detection_linear2", dtype='float32')) 
YOLO.add(tf.keras.layers.Reshape((7, 7, 30), name = 'output', dtype='float32'))

def yolo_multitask_loss(y_true, y_pred): # 커스텀 손실함수. 배치 단위로 값이 들어온다
    
    # YOLOv1의 Loss function은 3개로 나뉜다. localization, confidence, classification
    # localization은 추측한 box랑 ground truth box의 오차
    
    batch_loss = 0
    count = len(y_true)
    for i in range(0, len(y_true)) :
        y_true_unit = tf.identity(y_true[i])
        y_pred_unit = tf.identity(y_pred[i])
        
        y_true_unit = tf.reshape(y_true_unit, [49, 25])
        y_pred_unit = tf.reshape(y_pred_unit, [49, 30])
        
        loss = 0
        
        for j in range(0, len(y_true_unit)) :
            # pred = [1, 30], true = [1, 25]
            
            bbox1_pred = tf.identity(y_pred_unit[j][:4])
            bbox1_pred_confidence = tf.identity(y_pred_unit[j][4])
            bbox2_pred = tf.identity(y_pred_unit[j][5:9])
            bbox2_pred_confidence = tf.identity(y_pred_unit[j][9])
            class_pred = tf.identity(y_pred_unit[j][10:])
            
            bbox_true = tf.identity(y_true_unit[j][:4])
            bbox_true_confidence = tf.identity(y_true_unit[j][4])
            class_true = tf.identity(y_true_unit[j][5:])
            
            # IoU 구하기
            # x,y,w,h -> min_x, min_y, max_x, max_y로 변환
            box_pred_1_np = bbox1_pred.numpy()
            box_pred_2_np = bbox2_pred.numpy()
            box_true_np   = bbox_true.numpy()

            box_pred_1_area = box_pred_1_np[2] * box_pred_1_np[3]
            box_pred_2_area = box_pred_2_np[2] * box_pred_2_np[3]
            box_true_area   = box_true_np[2]  * box_true_np[3]

            box_pred_1_minmax = np.asarray([box_pred_1_np[0] - 0.5*box_pred_1_np[2], box_pred_1_np[1] - 0.5*box_pred_1_np[3], box_pred_1_np[0] + 0.5*box_pred_1_np[2], box_pred_1_np[1] + 0.5*box_pred_1_np[3]])
            box_pred_2_minmax = np.asarray([box_pred_2_np[0] - 0.5*box_pred_2_np[2], box_pred_2_np[1] - 0.5*box_pred_2_np[3], box_pred_2_np[0] + 0.5*box_pred_2_np[2], box_pred_2_np[1] + 0.5*box_pred_2_np[3]])
            box_true_minmax   = np.asarray([box_true_np[0] - 0.5*box_true_np[2], box_true_np[1] - 0.5*box_true_np[3], box_true_np[0] + 0.5*box_true_np[2], box_true_np[1] + 0.5*box_true_np[3]])

            # 곂치는 영역의 (min_x, min_y, max_x, max_y)
            InterSection_pred_1_with_true = [max(box_pred_1_minmax[0], box_true_minmax[0]), max(box_pred_1_minmax[1], box_true_minmax[1]), min(box_pred_1_minmax[2], box_true_minmax[2]), min(box_pred_1_minmax[3], box_true_minmax[3])]
            InterSection_pred_2_with_true = [max(box_pred_2_minmax[0], box_true_minmax[0]), max(box_pred_2_minmax[1], box_true_minmax[1]), min(box_pred_2_minmax[2], box_true_minmax[2]), min(box_pred_2_minmax[3], box_true_minmax[3])]

            # 박스별로 IoU를 구한다
            IntersectionArea_pred_1_true = 0

            # 음수 * 음수 = 양수일 수도 있으니 검사를 한다.
            if (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) >= 0 and (InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1) >= 0 :
                    IntersectionArea_pred_1_true = (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) * InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1

            IntersectionArea_pred_2_true = 0

            if (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) >= 0 and (InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1) >= 0 :
                    IntersectionArea_pred_2_true = (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) * InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1

            Union_pred_1_true = box_pred_1_area + box_true_area - IntersectionArea_pred_1_true
            Union_pred_2_true = box_pred_2_area + box_true_area - IntersectionArea_pred_2_true

            IoU_box_1 = IntersectionArea_pred_1_true/Union_pred_1_true
            IoU_box_2 = IntersectionArea_pred_2_true/Union_pred_2_true
                        
            responsible_IoU = 0
            responsible_box = 0
            responsible_bbox_confidence = 0
            non_responsible_bbox_confidence = 0

            # box1, box2 중 responsible한걸 선택(IoU 기준)
            if IoU_box_1 >= IoU_box_2 :
                responsible_IoU = IoU_box_1
                responsible_box = tf.identity(bbox1_pred)
                responsible_bbox_confidence = tf.identity(bbox1_pred_confidence)
                non_responsible_bbox_confidence = tf.identity(bbox2_pred_confidence)
                                
            else :
                responsible_IoU = IoU_box_2
                responsible_box = tf.identity(bbox2_pred)
                responsible_bbox_confidence = tf.identity(bbox2_pred_confidence)
                non_responsible_bbox_confidence = tf.identity(bbox1_pred_confidence)
                
            # 1obj(i) 정하기(해당 셀에 객체의 중심좌표가 들어있는가?)
            obj_exist = tf.ones_like(bbox_true_confidence)
            if box_true_np[0] == 0.0 and box_true_np[1] == 0.0 and box_true_np[2] == 0.0 and box_true_np[3] == 0.0 : 
                obj_exist = tf.zeros_like(bbox_true_confidence) 
            
                        
            # 만약 해당 cell에 객체가 없으면 confidence error의 no object 파트만 판단. (label된 값에서 알아서 해결)
            # 0~3 : bbox1의 위치 정보, 4 : bbox1의 bbox confidence score, 5~8 : bbox2의 위치 정보, 9 : bbox2의 confidence score, 10~29 : cell에 존재하는 클래스 확률 = pr(class | object) 

            # localization error 구하기(x,y,w,h). x, y는 해당 grid cell의 중심 좌표와 offset이고 w, h는 전체 이미지에 대해 정규화된 값이다. 즉, 범위가 0~1이다.
            localization_err_x = tf.math.pow( tf.math.subtract(bbox_true[0], responsible_box[0]), 2) # (x-x_hat)^2
            localization_err_y = tf.math.pow( tf.math.subtract(bbox_true[1], responsible_box[1]), 2) # (y-y_hat)^2

            localization_err_w = tf.math.pow( tf.math.subtract(tf.sqrt(bbox_true[2]), tf.sqrt(responsible_box[2])), 2) # (sqrt(w) - sqrt(w_hat))^2
            localization_err_h = tf.math.pow( tf.math.subtract(tf.sqrt(bbox_true[3]), tf.sqrt(responsible_box[3])), 2) # (sqrt(h) - sqrt(h_hat))^2
            
            # nan 방지
            if tf.math.is_nan(localization_err_w).numpy() == True :
                localization_err_w = tf.zeros_like(localization_err_w, dtype=tf.float32)
            
            if tf.math.is_nan(localization_err_h).numpy() == True :
                localization_err_h = tf.zeros_like(localization_err_h, dtype=tf.float32)
            
            localization_err_1 = tf.math.add(localization_err_x, localization_err_y)
            localization_err_2 = tf.math.add(localization_err_w, localization_err_h)
            localization_err = tf.math.add(localization_err_1, localization_err_2)
            
            weighted_localization_err = tf.math.multiply(localization_err, 5.0) # 5.0 : λ_coord
            weighted_localization_err = tf.math.multiply(weighted_localization_err, obj_exist) # 1obj(i) 곱하기
            
            # confidence error 구하기. true의 경우 답인 객체는 1 * ()고 아니면 0*()가 된다. 
            # index 4, 9에 있는 값(0~1)이 해당 박스에 객체가 있을 확률을 나타낸거다. Pr(obj in bbox)
            
            class_confidence_score_obj = tf.math.pow(tf.math.subtract(responsible_bbox_confidence, bbox_true_confidence), 2)
            class_confidence_score_noobj = tf.math.pow(tf.math.subtract(non_responsible_bbox_confidence, tf.zeros_like(bbox_true_confidence)), 2)
            class_confidence_score_noobj = tf.math.multiply(class_confidence_score_noobj, 0.5)
            
            class_confidence_score_obj = tf.math.multiply(class_confidence_score_obj, obj_exist)
            class_confidence_score_noobj = tf.math.multiply(class_confidence_score_noobj, tf.math.subtract(tf.ones_like(obj_exist), obj_exist)) # 객체가 존재하면 0, 존재하지 않으면 1을 곱합
            
            class_confidence_score = tf.math.add(class_confidence_score_obj,  class_confidence_score_noobj) 
            
            # classification loss(10~29. 인덱스 10~29에 해당되는 값은 Pr(Classi |Object)이다. 객체가 cell안에 있을 때 해당 객체일 확률
            # class_true_oneCell는 진짜 객체는 1이고 나머지는 0일거다. 
            
            tf.math.pow(tf.math.subtract(class_true, class_pred), 2.0) # 여기서 에러
            
            classification_err = tf.math.pow(tf.math.subtract(class_true, class_pred), 2.0)
            classification_err = tf.math.reduce_sum(classification_err)
            classification_err = tf.math.multiply(classification_err, obj_exist)

            # loss합체
            loss_OneCell_1 = tf.math.add(weighted_localization_err, class_confidence_score)
            loss_OneCell = tf.math.add(loss_OneCell_1, classification_err)
            
            if loss == 0 :
                loss = tf.identity(loss_OneCell)
            else :
                loss = tf.math.add(loss, loss_OneCell)
        
        if batch_loss == 0 :
            batch_loss = tf.identity(loss)
        else :
            batch_loss = tf.math.add(batch_loss, loss)
        
    # 배치에 대한 loss 구하기
    count = tf.Variable(float(count))
    batch_loss = tf.math.divide(batch_loss, count)
    
    return batch_loss


################################################
################################################
################### 훈련 세팅 ###################
################################################
################################################



BATCH_SIZE = 64
EPOCH = 135

# "We continue training with 10−2 for 75 epochs, then 10−3 for 30 epochs, and finally 10−4 for 30 epochs" 구현
def lr_schedule(epoch, lr): # epoch는 0부터 시작
    if epoch >=0 and epoch < 75 :
        lr = 0.001 + 0.009 * (float(epoch)/(75.0)) # 가중치를 0.001 ~ 0.0075로 변경
        return lr
    elif epoch >= 75 and epoch < 105 :
        lr = 0.001
        return lr
    else : 
        lr = 0.0001
        return lr

# loss 제일 낮을 때 가중치 저장
filename = 'yolo.h5'

checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             verbose=1,            # 로그를 출력합니다
                             save_best_only=True   # 가장 best 값만 저장합니다
                            )
                            
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum = 0.9)

YOLO.compile(loss = yolo_multitask_loss, optimizer=optimizer, run_eagerly=True)

YOLO.summary()


################################################
################################################
##################### 훈련 #####################
################################################
################################################

YOLO.fit(train_image_dataset, train_label_dataset,
          batch_size=BATCH_SIZE,
          validation_data = (val_image_dataset, val_label_dataset),
          epochs=EPOCH,
          verbose=1,
          callbacks=[checkpoint, tf.keras.callbacks.LearningRateScheduler(lr_schedule)])


################################################
################################################
#################### 테스트 #####################
################################################
################################################

# 출력된 bbox 정보를 사진에 출력할 수 있게 처리
def process_bbox(x, y, bbox, image_size, classes_score, Classes_inDataSet) : 
    # size 처리
    bbox_x = ((32.0 * x) + (bbox[0] * 32.0)) * (image_size[0]/224.0) # 예를 들어 x = 0이면 0~32사이에 중점의 x좌표가 존재
    bbox_y = ((32.0 * y) + (bbox[1] * 32.0)) * (image_size[1]/224.0) # 예를 들어 x = 0이면 0~32사이에 중점의 x좌표가 존재
    bbox_w = bbox[2] * image_size[0] # 전체 이미지 대비 백분위
    bbox_h = bbox[3] * image_size[1] # 전체 이미지 대비 백분위
    
    min_x = int(bbox_x - bbox_w/2)
    min_y = int(bbox_y - bbox_h/2)
    max_x = int(bbox_x + bbox_w/2)
    max_y = int(bbox_y + bbox_h/2)
    
    idx_class_highest_score = np.argmax(classes_score)
    class_highest_score = classes_score[idx_class_highest_score] # 가장 높은 class score
    class_highest_score_name = Classes_inDataSet[idx_class_highest_score] # 가장 높은 score를 가진 class의 이름
    
    output_bbox = [min_x, min_y, max_x, max_y, class_highest_score, class_highest_score_name]

    return output_bbox # [x, y, w, h, class_highest_score, class_highest_score_name]로 구성된 list출력
    
def nms(bbox_list) : 
    nms_bbox_list = []
    for i in range(0, len(bbox_list)) :
        
        if bbox_list[i][4] > 0.5 : # class score가 0.5넘기는 것만 출력하기
            nms_bbox_list.append(bbox_list[i])
    
    return nms_bbox_list
 
def get_YOLO_output(YOLO, Image_path, Classes_inDataSet) : 
    
    image_cv = cv2.imread(Image_path)
    height, width,_ = image_cv.shape # 이미지 원래 사이즈를 얻는다. [w, h]
    image_size = [width, height]

    image_cv = cv2.resize(image_cv, (224, 224))/255
    image_cv = np.expand_dims(image_cv, axis = 0)

    image_cv = image_cv.astype('float32')

    YOLO_output = YOLO(image_cv)[0].numpy() # 계산의 간편함을 위해 numpy array로 변환. [1,7,7,30]으로 나오기 때문에 [7,7,30]으로 만들어줘야한다.
    
    bbox_list = []
    
    for y in range(0, 7) :
        for x in range(0, 7) :
            # bbox에 있는 20개의 클래스 스코어
            bbox1_class_score = YOLO_output[y][x][10:] * YOLO_output[y][x][4]
            bbox2_class_score = YOLO_output[y][x][10:] * YOLO_output[y][x][9]
        
            # bbox의 사이즈
            bbox1 = YOLO_output[y][x][0:4]
            bbox2 = YOLO_output[y][x][5:9]
            
            # 24 -> 6(box info + 가장 높게 나온 클래스 prob + 가장 높게 나온 클래스의 idx)개로 처리
            # opencv는 min_x, min_y, max_x, max_y를 원하니 x, y, w, h를 min, max 좌표로 변환
            process_bbox1 = process_bbox(x, y, bbox1, image_size, bbox1_class_score, Classes_inDataSet)
            process_bbox2 = process_bbox(x, y, bbox2, image_size, bbox2_class_score, Classes_inDataSet)
            
            bbox_list.append(process_bbox1)
            bbox_list.append(process_bbox2)
    
    nms_bbox_list = nms(bbox_list)
    # nms_bbox_list = bbox_list
    
    im_read = cv2.imread(Image_path)

    for i in range(0, len(nms_bbox_list)) :
        
        # rectangle함수를 위해 필요한 '박스의 최소 x,y 좌표'와 '박스의 최대 x,y좌표'리스트를 생성한다. 
        min_box = (nms_bbox_list[i][0], nms_bbox_list[i][1])
        max_box = (nms_bbox_list[i][2], nms_bbox_list[i][3])
        # 출력하기
        cv2.rectangle(im_read, min_box, max_box, (0, 255, 0), 1) # 박스 그리기
#         show_str = nms_bbox_list[i][5] + " : " + str(nms_bbox_list[i][4])
        show_str = nms_bbox_list[i][5] # 객체 이름만 표시
        
        # 글자 넣어주기
        text_min_box = (nms_bbox_list[i][0] + 2, nms_bbox_list[i][1] - 10)
        text_max_box = (nms_bbox_list[i][2], nms_bbox_list[i][1])
        
        cv2.rectangle(im_read, text_min_box, text_max_box, (0, 255, 0), -1) # 박스 그리기
        
        cv2.putText(im_read, show_str, (min_box[0] + 2, min_box[1] - 1), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 1)
    
    cv2.imwrite('output.jpg', im_read)



