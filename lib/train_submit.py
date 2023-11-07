import json
import os
import glob

path = '/home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_fulltrain/results_mot17halfval'

for p1 in glob.glob(path+'/*'):
    print(p1)
    f = open(p1, "r")
    l = []
    for line in f:
        l.append(','.join(line.split()[0].split(',')[:-3]) + ',1,-1,-1,-1')
    with open(p1, 'w') as fp:
        fp.write('\n'.join(l))


'''mot_json = json.load(open(path + 'mot17/annotations/train.json', 'r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = 'mot_train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']

max_img = 10000
max_ann = 2000000
max_video = 10

crowdhuman_json = json.load(open(path + 'crowdhuman/annotations/val.json', 'r'))
img_id_count = 0
for n, img in enumerate(crowdhuman_json['images']):
    img_id_count += 1
    img['file_name'] = 'crowdhuman_val/' + img['file_name']
    img['frame_id'] = 1
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video + n
    img_list.append(img)
    video_list.append({'id': max_video + n})'''

'''for i in range(len(crowdhuman_json['images'])):
    img_id = crowdhuman_json['images'][i]['id']
    crowdhuman_json['images'][i]['video_id'] = img_id
    crowdhuman_json['images'][i]['frame_id'] = 1
    crowdhuman_json['videos'].append({'id': img_id})'''

'''for i in range(len(crowdhuman_json['annotations'])):
    crowdhuman_json['annotations'][i]['track_id'] = i + 1'''
