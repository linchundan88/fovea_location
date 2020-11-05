import cv2
import torch
from torchvision import transforms

def img_to_tensor(img_file, image_shape=(299, 299)):
    if isinstance(img_file, str):
        image = cv2.imread(img_file)
    else:
        image = img_file

    # shape: height, width,  resize: width, height
    # if (image_shape is not None) and (image.shape[:2] != image_shape[:2]):
    #     image = cv2.resize(image, (image_shape[1], image_shape[0]))

    # transform1 = transforms.Compose([
    #     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    # ])
    #
    # image = transform1(image)
    # image = image.unsqueeze(0)

    from imgaug import augmenters as iaa
    imgaug_iaa = iaa.Sequential([
        iaa.Resize(size=(image_shape))  # (height, weight)
    ])
    image = imgaug_iaa(image=image)
    image = transforms.ToTensor()(image)
    tensor = torch.unsqueeze(image, dim=0)

    return tensor
