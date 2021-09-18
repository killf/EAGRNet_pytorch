import numpy as np
import cv2

import os


def process(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    os.makedirs(os.path.join(dst_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, "labels"), exist_ok=True)

    # 1.prepare
    id_map = open(os.path.join(src_folder, "CelebA-HQ-to-CelebA-mapping.txt")).readlines()
    id_map = [line.split() for line in id_map[1:] if line.strip()]
    id_map = {int(idx): {"origin_file": img_file} for (idx, _, img_file) in id_map}

    lines = open(os.path.join(src_folder, "list_eval_partition.txt")).readlines()
    lines = [line.split() for line in lines if line.strip()]
    flags = {line[0]: int(line[1]) for line in lines}

    for k in id_map:
        id_map[k]["flag"] = flags[id_map[k]["origin_file"]]

    mask_map = {}
    mask_folder = os.path.join(src_folder, "CelebAMask-HQ-mask-anno")
    for folder_name in os.listdir(mask_folder):
        folder = os.path.join(mask_folder, folder_name)
        if not os.path.isdir(folder):
            continue

        for file_name in os.listdir(folder):
            if not file_name.endswith(".png"):
                continue

            idx = int(file_name[:5])
            if idx not in mask_map:
                mask_map[idx] = {}

            mask_map[idx][file_name[6:-4]] = os.path.join(folder_name, file_name)

    label_names = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'occlusion']
    label_map = {name: idx for idx, name in enumerate(label_names)}

    # 2.copy
    train, val, test = [], [], []
    for idx, item in id_map.items():
        src_file = os.path.join(src_folder, "CelebA-HQ-img", f"{idx}.jpg")
        src_img = cv2.imread(src_file)

        dst_img = cv2.resize(src_img, (512, 512))

        dst_file = os.path.join(dst_folder, "images", f"{idx}.jpg")
        cv2.imwrite(dst_file, dst_img)

        mask = None
        for label_name in label_names:
            if label_name in mask_map[idx]:
                m = cv2.imread(os.path.join(mask_folder, mask_map[idx][label_name]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros_like(m)
                mask[m != 0] = label_map[label_name]

        if mask is not None:
            line = f"{os.path.join('images', f'{idx}.jpg')},{os.path.join('labels', f'{idx}.png')}"
            label_file = os.path.join(dst_folder, "labels", f"{idx}.png")
            cv2.imwrite(label_file, mask)
        else:
            line = f"{os.path.join('images', f'{idx}.jpg')}"

        if item["flag"] == 0:
            train.append(line)
        elif item["flag"] == 1:
            val.append(line)
        else:
            test.append(line)

        print(f"{idx + 1}/{len(id_map)}", end="\r", flush=True)

    # 3.write
    open(os.path.join(dst_folder, "train.txt"), "w").write("\n".join(train))
    open(os.path.join(dst_folder, "val.txt"), "w").write("\n".join(val))
    open(os.path.join(dst_folder, "test.txt"), "w").write("\n".join(test))
    open(os.path.join(dst_folder, "label.txt"), "w").write(",".join(label_names))

    print("Complete!")


if __name__ == '__main__':
    process("/data/face/parsing/dataset/CelebAMask-HQ", "/data/face/parsing/dataset/CelebAMask-HQ_processed3")
