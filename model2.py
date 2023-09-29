import torch
import torch.nn as nn
import torch.functional as F

import matplotlib.pyplot as plt

from torchsummary import summary
import torchvision.models as models
from  torchvision.models import  densenet121 , DenseNet161_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Convnext(nn.Module):
    def __init__(self, num_classes=3):
        super(Convnext, self).__init__()
        #https://pytorch.org/vision/main/models/convnext.html
        self.convnext = models.convnext_tiny(pretrained = True)
        
        # for param in self.convnext.parameters():
        #     param.requires_grad = False
                
        
        self.convnext.classifier[-1] = nn.Linear(self.convnext.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        return self.convnext(x)




class Densenet(nn.Module):
    def __init__(self, num_classes=3):
        super(Densenet, self).__init__()
        #https://pytorch.org/vision/main/models/densenet.html
        self.densenet = models.densenet161(weights = DenseNet161_Weights.IMAGENET1K_V1)
        
        # for param in self.densenet.parameters():
        #     param.requires_grad = False
                
        
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)




class Efficientnet(nn.Module):
    def __init__(self, num_classes=3):
        super(Efficientnet, self).__init__()
        #https://pytorch.org/vision/stable/models/efficientnet.html
        self.efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # for param in self.efficientnet.parameters():
        #     param.requires_grad = False
                
        
        self.efficientnet.classifier[-1] = nn.Linear(self.efficientnet.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)


class Googlenet(nn.Module):
    def __init__(self, num_classes=3):
        super(Googlenet, self).__init__()
        #https://pytorch.org/vision/stable/models/googlenet.html
        self.googlenet = models.googlenet(weights='IMAGENET1K_V1')
        
        # for param in self.googlenet.parameters():
        #     param.requires_grad = False
                
        
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)
    
    def forward(self, x):
        if self.training:        
            return self.googlenet(x)
        else : 
            return self.googlenet(x)



class Mobilenetv3(nn.Module):
    def __init__(self, num_classes=3):
        super(Mobilenetv3, self).__init__()
        #https://pytorch.org/vision/main/models/mobilenetv3.html
        self.mobilenetv3 = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        # for param in self.mobilenetv3.parameters():
        #     param.requires_grad = False
                
        
        self.mobilenetv3.classifier[-1] = nn.Linear(self.mobilenetv3.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        return self.mobilenetv3(x)




class Mnasnet(nn.Module):
    def __init__(self, num_classes=3):
        super(Mnasnet, self).__init__()
        # https://pytorch.org/vision/main/models/mnasnet.html
        self.mnasnet = models.mnasnet0_75(weights='IMAGENET1K_V1')
        
        # for param in self.mnasnet.parameters():
        #     param.requires_grad = False
                
        
        self.mnasnet.classifier[-1] = nn.Linear(self.mnasnet.classifier[-1].in_features, num_classes)
    
    def forward(self, x):
        return self.mnasnet(x)




class SwinTransformer(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformer, self).__init__()
        # https://pytorch.org/vision/stable/models/swin_transformer.html
        self.swin = models.swin_t(weights='IMAGENET1K_V1')
        
        # for param in self.swin.parameters():
        #     param.requires_grad = False
                
        
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)
    
    def forward(self, x):
        return self.swin(x)






class Shufflenet(nn.Module):
    def __init__(self, num_classes=3):
        super(Shufflenet, self).__init__()
        # https://pytorch.org/vision/stable/models/shufflenetv2.html
        self.shufflenet = models.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')
        
        # for param in self.shufflenet.parameters():
        #     param.requires_grad = False
                
        
        self.shufflenet.fc = nn.Linear(self.shufflenet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.shufflenet(x)



class VisionTransformer(nn.Module):
    def __init__(self, num_classes=3):
        super(VisionTransformer, self).__init__()
        # https://pytorch.org/vision/stable/models/vision_transformer.html
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1')
        
        # for param in self.vit.parameters():
        #     param.requires_grad = False
                
        
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)




class WideRes(nn.Module):
    def __init__(self, num_classes=3):
        super(WideRes, self).__init__()
        # https://pytorch.org/vision/stable/models/wide_resnet.html
        self.wide_res = models.wide_resnet50_2(weights=None)
        
        # for param in self.wide_res.parameters():
        #     param.requires_grad = False
                
        
        self.wide_res.fc = nn.Linear(self.wide_res.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.wide_res(x)
    

    
if '__main__' == __name__:
    img = torch.rand(size=(1 ,3,224,224))
    model = WideRes()
    print(model(img))
    # summary(model.to(device) ,input_size= (3,224,224))