import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2

BATCH_SIZE = 8
EPOCH = 100
IMAGE_SIZE = np.array([676, 380, 3])
CSV_PATH = 'data/train_bbox.csv'
x_train_path = './data/train/*.jpg'
INPUT_SHAPE = (224, 224, 3)

# CSV: image 	xmin	ymin	xmax	ymax
def load_csv(csv_path: str)->pd.DataFrame:
    csv = pd.read_csv(csv_path)
    return csv

# USEAGE: show_bbox(csv['image'][0], cv2.imread('./data/train/' + csv['image'][0]), csv['xmin'][0], csv['ymin'][0], csv['xmax'][0], csv['ymax'][0])
def show_bbox(title: str, image: np.mat, xmin: float, ymin: float, xmax: float, ymax: float)->None:
    print('[show_bbox()]')

    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
    cv2.putText(image, title, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# CSV: image 	xmin	ymin	xmax	ymax
# return -> label: np.ndarray(7, 7, 25)
def cell_labeling(csv: pd.DataFrame, idx: int)->np.ndarray:
    # print('[cell_labeling()]')
    # csv->image  	xmin	ymin	xmx     ymax
    label = np.zeros((7, 7, 30), dtype = float)
    # label = np.zeros((7, 7, 25), dtype = float)
    ### Convert into 244 x 244 ###
    xmin = float((INPUT_SHAPE[0] / IMAGE_SIZE[0]) * csv['xmin'][idx])
    ymin = float((INPUT_SHAPE[1] / IMAGE_SIZE[1]) * csv['ymin'][idx])
    xmax = float((INPUT_SHAPE[0] / IMAGE_SIZE[0]) * csv['xmax'][idx])
    ymax = float((INPUT_SHAPE[1] / IMAGE_SIZE[1]) * csv['ymax'][idx])

    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = float((xmax - xmin) / INPUT_SHAPE[0])
    h = float((ymax - ymin) / INPUT_SHAPE[1])
    # print('x: {}, y: {}, w: {}, h: {}'.format(x, y, w, h))

    ### find out which cell (x, y) is in ###
    ### 7 x 7 grid ###
    cell_x = int(x / (INPUT_SHAPE[0] / 7))
    cell_y = int(y / (INPUT_SHAPE[1] / 7))
    center_x = float((x - cell_x * (INPUT_SHAPE[0] / 7)) / (INPUT_SHAPE[0] / 7))
    center_y = float((y - cell_y * (INPUT_SHAPE[1] / 7)) / (INPUT_SHAPE[1] / 7))
    # print('cell_x: {},  cell_y: {}'.format(cell_x, cell_y))

    label[cell_y, cell_x, 0] = center_x # center_x
    label[cell_y, cell_x, 1] = center_y # center_y
    label[cell_y, cell_x, 2] = w # width
    label[cell_y, cell_x, 3] = h # height
    label[cell_y, cell_x, 4] = 1.0 # confidence
    label[cell_y, cell_x, 5] = 1 # class 1개밖에 없음;;
    # print(label.shape)
    return label

def dataset(csv: pd.DataFrame, img_path_list: list)->tuple:
    print('[dataset()]')
    image_dataset = []
    label_dataset = []
    for i in tqdm(range(0, len(img_path_list))):
        # RGB(0~255) -> RGB(0~1)
        image = cv2.resize(cv2.imread(img_path_list[i]), (INPUT_SHAPE[0], INPUT_SHAPE[1])) / 255.0
        label = cell_labeling(csv,i)
        image_dataset.append(image)
        label_dataset.append(label)
    image_dataset = np.array(image_dataset, dtype="object")
    label_dataset = np.array(label_dataset, dtype="object")

    image_dataset = np.reshape(image_dataset, (-1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])).astype(np.float32)
    label_dataset = np.reshape(label_dataset, (-1, 7, 7, 30)).astype(np.float32)
    # label_dataset = np.reshape(label_dataset, (-1, 7, 7, 25)).astype(np.float32)
    print('[image_dataset.shape: {},    label_dataset.shape: {}]'.format(image_dataset.shape, label_dataset.shape))
    return image_dataset, tf.convert_to_tensor(label_dataset, dtype=tf.float32)



def LOSS(true_y, pred_y): # Loss Function; loss per batch
    loss=0
    for i in range(0, len(true_y)):
        pass
    return loss

def yolo_loss(y_true, y_pred): # 커스텀 손실함수. 배치 단위로 값이 들어온다
    
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



if __name__ == '__main__':
    print('[main()] This is api.py')