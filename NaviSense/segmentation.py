import _init_paths
import numpy as np
import PIDNet.models #need to locate!!
import torch
import torch.nn.functional as F

#variable setup
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]


pretrain_path_s = './PIDNet/output/custom/pidnet_small_custom/best.pt'

#Transfrom image to match traning data
def input_transform(image):
    #resize to match the traning data
    #image = image[:,0:960]
    
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

#Load the model
def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

#Initilize model
def init_model():
    
    ModelSize = 'pidnet-s'
    pretrain_path = pretrain_path_s
    model = PIDNet.models.pidnet.get_pred_model(ModelSize, 3) #select output here
    model = load_pretrained(model, pretrain_path).cuda()
    model.eval()
    return model

#Segmentatin function
def segmentationfun(model, img):
    with torch.no_grad():
        device = torch.device('cuda')
        model.to(device)
        
        img = input_transform(img)
        sv_img = np.zeros_like(img).astype(np.uint8)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        pred = model(img)
        pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()    
        for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
    
        #Transform to cv2 format
        sv_img = sv_img/255.0    
    return sv_img
