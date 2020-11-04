import cv2
import torch
from torchvision import transforms

def preprocess_image(img_file, image_shape=(299, 299)):
    image = cv2.imread(img_file)
    # shape: height, width,  resize: width, height
    if (image_shape is not None) and (image.shape[:2] != image_shape[:2]):
        image = cv2.resize(image, (image_shape[1], image_shape[0]))
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    image = transform1(image)

    '''
    img_1 = img1.numpy()*255
    img_1 = img_1.astype('uint8')
    img_1 = np.transpose(img_1, (1,2,0))
    cv2.imshow('img_1', img_1)
    cv2.waitKey()
    '''
    image = image.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = image.to(device)

    return inputs
