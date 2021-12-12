import os

import numpy as np
import requests
from imageio import imread, imsave

from parser_service.service import FaceParserService
from parser_service.validators import InferenceData


def debug_direct():
    face_swap_service = FaceParserService()
    #
    results = face_swap_service.predict([item1, item2])
    for index, result in enumerate(results):
        img = np.frombuffer(result.data, dtype="uint8").reshape(512, 512, 4)
        imsave(f'output/head_{index}.png', img)
    print('ok')


def debug_http():
    face_swap_service = FaceParserService(serving=False)
    face_swap_service.start_dev_server()

    response = requests.post("http://127.0.0.1:5000/predict", json=[item1, item2])
    print(response.text)
    print('ok')


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    crop1 = imread('../data/musk.jpg').astype("uint8")
    crop2 = imread('data/emma_full.jpg').astype("uint8")
    #
    item1 = InferenceData(image=crop1)
    item2 = InferenceData(image=crop2)

    debug_direct()
    # debug_http()
    print('ok')
