# -*- encoding: utf-8 -*-
# Date: 18/Jan/2022
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""
Description: Yolov5 pretrained weights download
"""

import os
import sys

yolov5_path = r'../yolov5-master/'  # Please change to your yolov5-master path
sys.path.append(yolov5_path)


from utils.downloads import attempt_download


def main():
    for x in ['s', 'm', 'l', 'x']:
        file = os.path.join(r'.\weights', f'yolov5{x}.pt')
        attempt_download(file)


if __name__ == "__main__":
    main()
