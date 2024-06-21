import torch
from torchvision import transforms
import cv2
class MyData(torch.utils.data.Dataset):
    def __init__(self,  label_path, transforms=None):
        #label_path: image label path
        self.transforms = transforms
        self.label_path = label_path
        self.imgs = []
    
        with open(label_path, 'r') as fp:
            for line in fp:
                line = line.strip()
                sample = line.split(' ')# exp: ta.jpg ta_gt.png 1
                self.imgs.append((sample[0], sample[1], sample[2]))
                
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        tamper_img, tamper_mask, label_ta = self.imgs[index]
        #read tampered_image
        taimg = cv2.imread(tamper_img)
        #read mask_image
        mask = cv2.imread(tamper_mask, cv2.IMREAD_GRAYSCALE)
        #get tampered_image size
        h = taimg.shape[0]
        w = taimg.shape[1]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask_trans = transforms.Compose([
            transforms.ToTensor(),
              transforms.Resize([h,w]),])
        else:
            mask_trans = transforms.Compose([
            transforms.ToTensor(),
              ])
        b,g,r = cv2.split(taimg)
        taimg = cv2.merge([r,g,b])

        transform_def = transforms.Compose([
            transforms.ToTensor(),
              ])
        return transform_def(taimg), label_ta, tamper_img, h, w,mask_trans(mask)




