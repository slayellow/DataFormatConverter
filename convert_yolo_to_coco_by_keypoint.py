import json
import os
import cv2

prefixs = ['/home/a2mind/Data/Ikksu/CCTV/cam2/']

for prefix in prefixs:
    json_path = os.path.join(prefix, 'annotations')
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
    json_file = os.path.join(json_path, 'person_keypoints_result.json')

    print('=' * 50)
    print('COCO KeyPoint Format Start!')
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
    coco_json["categories"] = [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose","left_eye","right_eye","left_ear","right_ear",
                    "left_shoulder","right_shoulder","left_elbow","right_elbow",
                    "left_wrist","right_wrist","left_hip","right_hip",
                    "left_knee","right_knee","left_ankle","right_ankle"
                ],
                "skeleton": [
                    [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                    [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
                ]
            }
        ]

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

                num_kpts = 0
                kpts = det[6:]
                steps = 3

                keypoints = []
                for kid in range(len(kpts) // steps):
                    x_coord, y_coord = float(kpts[steps * kid]), float(kpts[steps * kid + 1])
                    x_coord = x_coord * width
                    y_coord = y_coord * height
                    if steps == 3:
                        conf = float(kpts[steps * kid + 2])
                        if conf < 0.5:
                            keypoints.extend([x_coord, y_coord, 0])
                        else:
                            num_kpts += 1
                            keypoints.extend([x_coord, y_coord, 2])

                part_annotations = dict()
                part_annotations['id'] = annotations_id
                part_annotations['image_id'] = idx
                part_annotations['category_id'] = 1
                part_annotations['segmentation'] = []

                if num_kpts == 0:
                    part_annotations['area'] = 0.0
                    part_annotations['bbox'] = [0.0, 0.0, 0.0, 0.0]
                else:
                    part_annotations['area'] = area
                    part_annotations['bbox'] = [(xmin + xmax) / 2,
                                                (ymin + ymax) / 2,
                                                (xmax - xmin),
                                                (ymax - ymin)]
                part_annotations['iscrowd'] = 0
                part_annotations['attributes'] = { 'occluded': False }                
                part_annotations['keypoints'] = keypoints
                part_annotations['num_keypoints'] = num_kpts

                annotations_id += 1
                coco_json['annotations'].append(part_annotations)

with open(json_file, 'w') as f:
    json.dump(coco_json, f, ensure_ascii=False, indent=4)