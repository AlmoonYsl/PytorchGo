import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('0', '1', '2', '3', '4', '5', '6',
           '7', '8', '9')


def reverse(img):
    h, w = img.shape
    dst = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            dst[i, j] = 255 - img[i, j]
    return dst


if __name__ == '__main__':
    net = torch.load('./net/net.pkl')
    img = cv2.imread('5.jpg', 0)
    img = cv2.resize(img, (28, 28))
    plt.imshow(img)
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    net.to(device)
    net.eval()
    plt.show()
    img = img.to(device)
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%2s' % classes[predicted[j]] for j in range(1)))
