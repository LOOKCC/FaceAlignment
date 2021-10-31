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
        w = x2-x1
        h = y2-y1
        new_x1 = x1 - int(w*((size-1)/2))
        new_x2 = x2 + int(w*((size-1)/2))
        new_y1 = y1 - int(h*((size-1)/2))
        new_y2 = y2 + int(h*((size-1)/2))

        # assert new_x1 > big_bbox[0]
        # assert new_y1 > big_bbox[1]
        # assert new_x2 < big_bbox[2]
        # assert new_y2 < big_bbox[3]
        if new_x1 > big_bbox[0] and new_y1 > big_bbox[1] and new_x2 < big_bbox[2] and new_y2 < big_bbox[3]:
            com_new_x1 = new_x1 - big_bbox[0]
            com_new_x2 = new_x2 - big_bbox[0]
            com_new_y1 = new_y1 - big_bbox[1]
            com_new_y2 = new_y2 - big_bbox[1]

            bboxs.append([new_x1, new_y1, new_x2, new_y2])
            lm5 = data['landmarks']
            new_lm5 = []
            
            for j in range(5):
                # new_lm5.append(np.asarray([lm5[j*2+1] - new_x1, lm5[j*2] - new_y1]))
                new_lm5.append(np.asarray([lm5[j*2] - new_y1, lm5[j*2+1] - new_x1]))
            
            lm5s.append(new_lm5)
            face_image = frames[i][com_new_y1:com_new_y2,com_new_x1:com_new_x2]
            face_images.append(face_image)
        else:
            pass
        
    face_images = [np.asarray(x) for x in face_images]
    bboxs = np.array([np.asarray(x) for x in bboxs])
    lm5s = np.array([np.asarray(x) for x in lm5s])
    images = face_alogn.retinaface(lm5s, bboxs, face_images)

    return images

def check(json_name, size):
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
    # size = 1.3
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

    # image1 = cv2.imread('/data/fanglingfei/dataset/Celeb-DF_consecutive_retina_face/Celeb-synthesis/id8_id2_0007/id8_id2_0007_280.png')
    # json1 = json.load(open('/data/fanglingfei/dataset/Celeb-DF_consecutive_retina_face/Celeb-synthesis/id8_id2_0007/id8_id2_0007_280.json', 'r'))

    # cv2.imwrite('./pics/ori.png', image1)
    image1 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_424.png')
    image2 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_425.png')
    image3 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_426.png')
    image4 = cv2.imread('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_427.png')

    json1 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_424.json', 'r'))
    json2 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_425.json', 'r'))
    json3 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_426.json', 'r'))
    json4 = json.load(open('/data/fanglingfei/dataset/faceforensics_c23_consecutive_retina_face/VideoData/manipulated_sequences/Deepfakes/c23/videos/412_274/412_274_427.json', 'r'))

    size = 1.3

    frames = [image1, image2, image3, image4]
    jsons = [json1, json2, json3, json4]

    # frames = [image1]
    # jsons = [json1]
    
    images = align(frames, jsons, size)
    # image = images[0]
    # cv2.imwrite('./pics/test.png', image)