#get image label, save as xxx.txt
#image label example: ta.jpg ta_gt.png 1
import os

img_path = '../datasets/CASIA/'
log_file = open('casia.txt', 'w', encoding='utf-8')


def get_GTname(path):
    if "CASIA" in path:
        gt_path = path.replace("CASIA", "CASIA_GT")
        gt_path = gt_path.replace(".jpg", "_gt.png")
        return gt_path
    if "Columbia" in path:
        gt_path = path.replace("Columbia", "Columbia_GT")
        gt_path = gt_path.replace(".tif", "_gt.png")
        return gt_path
    if "coverage" in path:
        gt_path = path.replace("coverage_100", "coverage_GT")
        gt_path = gt_path.replace("t.tif", "forged.tif")
        return gt_path
    if "DSO" in path:
        gt_path = path.replace("DSO_crop", "DSO_GT_crop")
        gt_path = gt_path.replace(".png", "_gt.png")
        return gt_path
    if "IMD" in path:
        gt_path = path.replace("IMD_crop", "IMD_GT_crop")
        gt_path = gt_path.replace(".jpg", ".png")
        return gt_path

for file in os.listdir(img_path):
  if (os.path.isfile(os.path.join(img_path,file))==True) and file !='casia.txt':
    file_path = os.path.join(img_path,file)
    gt_path = get_GTname(file_path)
    
    log_file.write(file_path)
    log_file.write(' ')
    log_file.write(gt_path)
    log_file.write(' ')
    log_file.write('1')
    log_file.write('\n')
log_file.close()
