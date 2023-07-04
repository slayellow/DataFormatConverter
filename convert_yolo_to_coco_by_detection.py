import json
import os
import cv2

prefixs = ['/home/a2mind/Data/Ikksu/CCTV/cam2/']
classes = ['person', 'safe', 'danger', 'stair']


for prefix in prefixs:
    json_path = os.path.join(prefix, 'annotations')
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
    json_file = os.path.join(json_path, 'detection_result.json')

    print('=' * 50)
    print('COCO Format Start!')
    print('=' * 50)

    coco_json = {}
    # License
    coco_json["licenses"] =[
                            {
                            "name": "",
                            "id": 0,
                            "url": ""
                            }
                            ]
    # Info
    coco_json["info"] = {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""
        }
    # Categories
    coco_json['categories'] = list()
    for idx, cls in enumerate(classes):
        category = {
                    "id": idx + 1,
                    "name": cls,
                    "supercategory": ""
                    }
        coco_json['categories'].append(category)

    image_list = list(sorted(os.listdir(os.path.join(prefix, 'images'))))

    coco_json["images"] = list()
    coco_json["annotations"] = list()

    annotations_id = 1
    for idx, img_path in enumerate(image_list):
        idx += 1
        print(idx, img_path)
        image = cv2.imread(os.path.join(prefix, 'images', img_path))
        height, width = image.shape[:2]

        coco_json["images"].append(
            {
                "id": idx,
                "width": width,
                "height": height,
                "file_name": img_path,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            }
        )

        lbl_path = os.path.join(prefix, 'labels', img_path.replace('.jpg', '.txt'))
        if not os.path.exists(lbl_path):
            continue
        else: 
            for line in open(lbl_path, 'r'):
                det = line.split()
                cls = int(det[0])
                xmin = (float(det[1]) - 0.5 * float(det[3])) * width
                xmax = (float(det[1]) + 0.5 * float(det[3])) * width
                ymin = (float(det[2]) - 0.5 * float(det[4])) * height
                ymax = (float(det[2]) + 0.5 * float(det[4])) * height
                area = (xmax - xmin) * (ymax - ymin)

                part_annotations = dict()
                part_annotations['id'] = annotations_id
                part_annotations['image_id'] = idx
                part_annotations['category_id'] = cls + 1
                part_annotations['segmentation'] = []

                part_annotations['area'] = area
                part_annotations['bbox'] = [(xmin + xmax) / 2,
                                            (ymin + ymax) / 2,
                                            (xmax - xmin),
                                            (ymax - ymin)]
                
                part_annotations['iscrowd'] = 0
                part_annotations['attributes'] = { 'occluded': False }                
                annotations_id += 1
                coco_json['annotations'].append(part_annotations)

with open(json_file, 'w') as f:
    json.dump(coco_json, f, ensure_ascii=False, indent=4)