import re
import cv2
import os
import numpy as np
from tqdm import tqdm

# Total-Text To IC15

# F:\zzxs\Experiments\dl-data\TotalText\Groundtruth\Text\Legacy
# F:\zzxs\Experiments\dl-data\TotalText\Groundtruth\Text\Legacy\\txt_format\Test
root_path = '/home/zhangyangsong/OCR/totaltext_gt/txt/Train'
dest_path = '/home/zhangyangsong/OCR/totaltext_gt/txt/Train_change'
_indexes = os.listdir(root_path)

withTranscription = True

def cvt_total_text():
    invalid_count = 0
    all_count = 0
    for index in tqdm(_indexes, desc='convert labels'):
        if os.path.splitext(index)[1] != '.txt':
            continue
        anno_file = os.path.join(root_path, index)

        with open(anno_file, 'r+') as f:
            # lines是每个文件中包含的内容
            lines = [line for line in f.readlines() if line.strip()]
            single_list = []
            all_list = []
            try:
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    while not line.endswith(']'):
                        print('concat in ', index)
                        i = i + 1
                        line = line + ' ' + lines[i].strip()
                    i += 1

                    if line[-3] == '#':
                        invalid_count += 1
                    all_count += 1

                    parts = line.split(',')
                    xy_list = []
                    for a, part in enumerate(parts):
                        if a > 1:
                            break
                        piece = part.strip().split(',')
                        numberlist = re.findall(r'\d+', piece[0])
                        xy_list.extend(numberlist)

                    length = len(xy_list)
                    n = int(length / 2)
                    x_list = xy_list[:n]
                    y_list = xy_list[n:]
                    single_list = [None] * (len(x_list) + len(y_list))
                    single_list[::2] = x_list
                    single_list[1::2] = y_list

                    if withTranscription:
                        parts = line.split('\'')
                        transcription = parts[-2]
                        single_list.append(transcription)
                    all_list.append(single_list)

            except Exception as e:
                print('error: ', index)
        dest_file = os.path.join(dest_path, index)
        with open(dest_file, 'w') as w:
            for all_list_piece in all_list:
                w.write(','.join(all_list_piece))
                w.write('\n')
    print('### count: ', invalid_count)
    print('All count: ', all_count)
    print('rate: ', float(invalid_count)/all_count)

if __name__ == '__main__':
    cvt_total_text()