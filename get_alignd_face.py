import numpy as np
import cv2
import json
from faster_crop_align_xray import FasterCropAlignXRay
import os
from tqdm import tqdm


face_alogn = FasterCropAlignXRay(224)

def align(frames, jsons, size):
    face_images = []
    bboxs = []
    lm5s = []
    for i, data in enumerate(jsons):
        org_bbox = data['ori_coordinate']
        big_bbox = data['coordinates']

        x1, y1, x2, y2 = org_bbox
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        
        
        w = x2-x1
        h = y2-y1
        new_x1 = x1 - int(w*((size-1)/2))
        new_x2 = x2 + int(w*((size-1)/2))
        new_y1 = y1 - int(h*((size-1)/2))
        new_y2 = y2 + int(h*((size-1)/2))
        new_x1 = max(new_x1, 0)
        new_y1 = max(new_y1, 0)
        # print(new_x1, new_y1, new_x2, new_y2)
        # assert new_x1 > big_bbox[0]
        # assert new_y1 > big_bbox[1]
        # assert new_x2 < big_bbox[2]
        # assert new_y2 < big_bbox[3]
        if new_x1 >= big_bbox[0] and new_y1 >= big_bbox[1] and new_x2 <= big_bbox[2] and new_y2 <= big_bbox[3]:
            # print('here')
            com_new_x1 = new_x1 - big_bbox[0]
            com_new_x2 = new_x2 - big_bbox[0]
            com_new_y1 = new_y1 - big_bbox[1]
            com_new_y2 = new_y2 - big_bbox[1]

            bboxs.append([new_x1, new_y1, new_x2, new_y2])
            lm5 = data['landmarks']
            new_lm5 = []

            lm5 = [max(x, 0) for x in lm5]            
            for j in range(5):
                # 这种情况是对的
                # print(np.asarray([lm5[j*2] - big_bbox[1], lm5[j*2+1] - big_bbox[0]]))
                new_lm5.append(np.asarray([lm5[j*2] - new_y1, lm5[j*2+1] - new_x1]))
                # [57 65]
                # [86 63]
                # [71 81]
                # [61 95]
                # [85 93]
                # 在原始的代码中的结果是
                # [[56.850048, 66.091324],
                # [84.83173 , 64.05539 ],
                # [70.842224, 81.82417 ],
                # [61.36377 , 95.456154],
                # [84.85769 , 93.667595]]
                # 结果对上了
                # new_lm5.append(np.asarray([lm5[j*2+1] - new_x1, lm5[j*2] - new_y1])) # 错的 错的 错的
                
            
            lm5s.append(new_lm5)
            face_image = frames[i][com_new_y1:com_new_y2,com_new_x1:com_new_x2]
            face_images.append(face_image)
        else:
            pass
        
    face_images = [np.asarray(x) for x in face_images]
    bboxs = np.array([np.asarray(x) for x in bboxs])
    lm5s = np.array([np.asarray(x) for x in lm5s])
    # 这里可以根据True False 的结果选择直接放弃此样本，或者使用不对齐的版本
    if len(frames) == len(face_images):
        images = face_alogn.retinaface(lm5s, bboxs, face_images)
        return True, images
    else:
        return False, frames

def check(json_name, size=1.3):
    data = json.load(open(json_name, 'r'))
    org_bbox = data['ori_coordinate']
    big_bbox = data['coordinates']
    x1, y1, x2, y2 = org_bbox
    w = x2-x1
    h = y2-y1
    new_x1 = x1 - int(w*((size-1)/2))
    new_x2 = x2 + int(w*((size-1)/2))
    new_y1 = y1 - int(h*((size-1)/2))
    new_y2 = y2 + int(h*((size-1)/2))

    if new_x1 > big_bbox[0] and new_y1 > big_bbox[1] and new_x2 < big_bbox[2] and new_y2 < big_bbox[3]:
        return True
    else:
        return False



def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs: 
            fullname = os.path.join(root, f)
            yield fullname




if __name__ == "__main__":
    # 首先测试有多少可行
    size = 1.3
    # yes = 0
    # no = 0
    # all_file = []
    # for file in findAllFile('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/'):
    #     if file[-4:] == 'json':
    #         all_file.append(file)
    # print(len(all_file))
    # for file in tqdm(all_file):
    #     if check(file, size):
    #         yes += 1
    #     else:
    #         no += 1
    # print(yes)
    # print(no)
    # 结果：1.5倍 一共752189 可行677691 不行74498
    # 1.3倍 752189 706129 46060

    # 检查连续四帧的可行性
    total = 0
    yes = 0
    no = 0
    with open('/data/fanglingfei/workspace/universal5/data/faceforensics_c23_consecutive_faces_4_wza236/train_annotations/faceforensics_c23_train.txt', 'r') as f:
        for line in tqdm(f.readlines()):
            total += 1
            image1, image2, image3, image4, label = line.split()
            json1 = image1.replace('.png', '.json')
            json2 = image2.replace('.png', '.json')
            json3 = image3.replace('.png', '.json')
            json4 = image4.replace('.png', '.json')
            if check(json1) and check(json2) and check(json3) and check(json4):
                yes += 1
            else:
                no +=1
    print(total, yes, no)
    # 142282 132834 9448










    # image1 = cv2.imread('/data/fanglingfei/dataset/Celeb-DF_consecutive_retina_face/Celeb-synthesis/id8_id2_0007/id8_id2_0007_280.png')
    # json1 = json.load(open('/data/fanglingfei/dataset/Celeb-DF_consecutive_retina_face/Celeb-synthesis/id8_id2_0007/id8_id2_0007_280.json', 'r'))

    # image1 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/original_sequences/actors/c23/videos/20__kitchen_pan/20__kitchen_pan_20.png')
    # json1 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/original_sequences/actors/c23/videos/20__kitchen_pan/20__kitchen_pan_20.json', 'r'))
    # print(json1)
    # cv2.imwrite('./pics/ori.png', image1)
    # image1 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_424.png')
    # image2 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_425.png')
    # image3 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_426.png')
    # image4 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_427.png')

    # json1 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_424.json', 'r'))
    # json2 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_425.json', 'r'))
    # json3 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_426.json', 'r'))
    # json4 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_427.json', 'r'))

    # size = 1.3

    # frames = [image1, image2, image3, image4]
    # jsons = [json1, json2, json3, json4]

    # # frames = [image1]
    # # jsons = [json1]
    
    # images = align(frames, jsons, size)
    # image = images[0]
    # cv2.imwrite('./pics/test.png', image)