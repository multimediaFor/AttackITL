import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from load_tdata_mask import MyData
from torchvision.utils import save_image

#OSN model
from models.scse import SCSEUnet

gpu_ids = '0, 1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
device = torch.device('cuda:0')

#image label path
#image label example: ta.jpg ta_gt.png 1
path = '../casia.txt'
datapath = '../datasets/CASIA/'
output_path = './FGSM_casia_0.02/'
if not os.path.exists(output_path):
            os.makedirs(output_path)

#set step_size eps
eps = 0.02

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


#load localizer OSN
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.name = 'detector'
        self.det_net = SCSEUnet(backbone_arch='senet154', num_channels=3)

    def forward(self, Ii):
        Mo = self.det_net(Ii)
        return Mo
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.save_dir = 'weights/'
        self.networks = Detector()
        self.gen = nn.DataParallel(self.networks).cuda()

    def forward(self, Ii):
        return self.gen(Ii)

    def load(self, path=''):
        self.gen.load_state_dict(torch.load(self.save_dir + path + '%s_weights.pth' % self.networks.name))


def attack(model, eps):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    #load dataset
    test_loader = DataLoader(dataset=MyData(path), batch_size=1, shuffle=False, num_workers=1)
    
    #tamper_image, tampered-1/authentic-0, image_path, height, width, mask
    for _,(orig_image,clss,name,h,w,ta_mask) in enumerate(test_loader):
        orig_image = orig_image.cuda()
        ta_mask = ta_mask.cuda()
        
        #save as jpeg format
        attack_path = name[0].replace(datapath,output_path)
        attack_path = attack_path.replace(".JPG",".jpg")#jpg
        attack_path = attack_path.replace(".tif",".jpg")#jpg
        attack_path = attack_path.replace(".TIF",".jpg")#jpg
        attack_path = attack_path.replace(".png",".jpg")#jpg
        print("attack:",attack_path)

        orig_image.requires_grad = True
        pred = model(orig_image)
        loss_func = torch.nn.BCELoss().cuda()
        loss = loss_func(pred, ta_mask)
        model.zero_grad()
        loss.backward()

        data_grad = orig_image.grad.data
        adv = fgsm_attack(orig_image, eps, data_grad)
        save_image(adv,attack_path)

        



if __name__ == "__main__":
    OSN = Model().cuda()
    OSN.load()
    OSN.eval()
    attack(OSN, eps)
