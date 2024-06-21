import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from load_tdata_mask import MyData
from torchvision.utils import save_image
import logging

#OSN model
from models.scse import SCSEUnet

gpu_ids = '0, 1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
device = torch.device('cuda:0')

#image label path
#image label example: ta.jpg ta_gt.png 1
log_path = './log.txt'
labels = '../casia.txt'
datapath = '../datasets/CASIA/'
output_path = './OPT_casia_0.003/'
if not os.path.exists(output_path):
            os.makedirs(output_path)

#set batch_size, learning_rate and epoch
batch = 1
learing_rates = [0.003]
temp_epoch = [30]

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

#define logger
def get_log(file_name):
    logger = logging.getLogger('train')  # set logger name
    logger.setLevel(logging.INFO)  #set logger level
    
    if not logger.handlers: 
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(file_name, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


#load dataset
train_ds= MyData(labels)
dataloader = torch.utils.data.DataLoader(train_ds,batch_size = batch,shuffle = False,
                                         num_workers=8)



def attack(model, learing_rate, epochs):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    for step,(orig_image,clss,name,h,w,ta_mask) in enumerate(dataloader):
        orig_image = orig_image.cuda()
        ta_mask = ta_mask.cuda()
        out_path = 'opt_' + str(learing_rate)+'_casia'
        
        #save as jpeg format
        attack_path = name[0].replace(datapath,out_path)
        attack_path = attack_path.replace(".JPG",".jpg")
        attack_path = attack_path.replace(".tif",".jpg")
        attack_path = attack_path.replace(".TIF",".jpg")
        attack_path = attack_path.replace(".png",".jpg")
        print(attack_path)

        #Optimization-based 
        if not(os.path.exists(attack_path)):
            orig_image.requires_grad = True # get input grad
            # set orig_image as the optimization parameter for the optimizer
            optimizer = torch.optim.Adam(params=[orig_image],lr=learing_rate)
            loss_func = torch.nn.BCELoss().cuda()
            for epoch in range(epochs):
                optimizer.zero_grad()
                tensor_orig_image = torch.clamp(orig_image, 0, 1)
                pred_mask = model(tensor_orig_image)
                target_label_index = torch.zeros(pred_mask.shape).cuda()
        
                loss = loss_func(pred_mask,target_label_index)
                loss.backward()
                optimizer.step() # Update parameters, i.e. update input examples
            
                #Save optimization information for each epoch
                logger = get_log(log_path) 
                logger.info("img:%d epoch:%d loss:%.4f"%(step, epoch, loss.item()))
                if epoch == epochs-1 :
                    save_image(tensor_orig_image,attack_path)
                    break

            
def main():
    OSN = Model().cuda()
    OSN.load()
    OSN.eval()
    for lr in learing_rates:
        attack(OSN, lr, temp_epoch)

if __name__ == "__main__":
    main()


