import os


def process_folder(src_folder, folder_name, target_name):
    lines = []
    for file_name in os.listdir(os.path.join(src_folder, folder_name)):
        if not file_name.endswith(".jpg"):
            continue

        file_name = file_name[:-4]
        lines.append(f"{folder_name}/{file_name}.jpg,{folder_name}/{file_name}.png")
    open(os.path.join(src_folder, target_name), "w").write("\n".join(lines))


def process(src_folder):
    process_folder(src_folder, "train", "train.txt")
    process_folder(src_folder, "test", "val.txt")
    open(os.path.join(src_folder, "label.txt"), "w").write("background,skin,left_eyebrow,right_eyebrow,left_eye,right_eye,nose,upper_lip,inner_mouth,lower_lip,hair")


if __name__ == '__main__':
    process("/data/face/parsing/dataset/ibugmask_release")
