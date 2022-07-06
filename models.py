
# import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F

class EfficientNet_b4(nn.Module):

  def __init__(self,pretrained):
    super(EfficientNet_b4,self).__init__()
    if pretrained is True:
      self.model = EfficientNet.from_pretrained('efficientnet-b0')
    else:
      self.model = EfficientNet.from_name('efficientnet-b0')
    self.l0 = nn.Linear(1280,168)
    self.l1 = nn.Linear(1280,11)
    self.l2 = nn.Linear(1280,7)
  
  def forward(self,inputs):
    bs,_,_,_=inputs.shape
    x=self.model.extract_features(inputs)
    x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
    l0 = self.l0(x)
    l1 = self.l1(x)
    l2 = self.l2(x)
    return l0,l1,l2
