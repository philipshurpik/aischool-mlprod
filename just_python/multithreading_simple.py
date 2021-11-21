import time
from threading import Thread
# import cv2
import numpy as np
import torch
import torchvision

model = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
model = model.eval().requires_grad_(False).to(torch.float32).to('cpu')

# cv2.setNumThreads(1)
torch.set_num_threads(1)
COUNT = 50000000


def resize(im_frames):
    # [cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LANCZOS4) for frame in im_frames]
    [model(torch.tensor(frames[0]).permute(2, 0, 1).div(255).unsqueeze(0))for frame in im_frames]


def run_single_thread(frames):
    start = time.time()
    resize(frames)
    end = time.time()
    print('Elapsed time single thread:', end - start)


def run_multi_thread(frames, cores=8):
    start = time.time()
    n = len(frames) // cores
    threads = [Thread(target=resize, args=(frames[n * i: n * (i + 1)],)) for i in range(cores)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    end = time.time()
    print(f'Elapsed time {cores} threads:', end-start)


if __name__ == '__main__':
    frames = np.random.randint(255, size=(80, 1280, 720, 3), dtype=np.uint8)
    run_single_thread(frames)
    run_multi_thread(frames)
