import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from torchvision import transforms as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

device =  'cuda' if torch.cuda.is_available() else 'cpu'


α = 1   # 内容损失权重
β = 1e3  # 风格损失权重
γ = 0
EPOCH = 100 # 迭代次数
Content_layer = 4
closs, sloss = [0.],[0.]
style_layer = 5
# 损失函数
mse_loss = nn.MSELoss(reduction='mean') 


# 图像预处理
transform = tf.Compose([
                tf.Resize((512,512)),
                tf.ToTensor(),
                tf.Normalize([0.485, 0.456, 0.406], [1, 1, 1]),
            ])
decode = tf.Compose([
                tf.Normalize([-0.485,-0.456,-0.406], [1, 1, 1]),                
                tf.Lambda(lambda x: x.clamp(0,1))
            ])
tensor2PIL = tf.ToPILImage()




style_img_path = '../input/styles/style/monet.jpg'  # 风格图
content_img_path = '../input/imagesdata/old.png'    # 内容图

style_img = Image.open(style_img_path)
content_img = Image.open(content_img_path)

style_img = transform(style_img).to(device)
content_img = transform(content_img).to(device)

# 获取网络某层的输出
def get_features(module, x, y):
    features.append(y)
    
# 计算格拉姆矩阵
def gram_matrix(x):
    # x = x.unsqueeze(0)
    b, c, h, w = x.size()
    F = x.view(b,c,h*w)
    # torch.bmm计算两个矩阵的矩阵乘法，维度必须是(batches, w, h)
    G = torch.bmm(F, F.transpose(1,2))/(h*w)
    return G

# 内容损失:
class content_loss(nn.Module):
    def __init__(self):
        super(content_loss, self).__init__()

    def forward(self, content, content_target):
        c_loss = mse_loss(content, content_target)
        return c_loss

# 风格损失:
class style_loss(nn.Module):
    def __init__(self):
        super(style_loss, self).__init__()

    def forward(self, gram_styles, gram_targets):
        s_loss = 0
        for i in range(5):
            # N = gram_styles[i].shape[-1]
            # M = style_features[i].shape[-1]
            s_loss += mse_loss(gram_styles[i],gram_targets[i])
        return s_loss

# # 平滑损失:
# class smooth_loss(nn.Module):
#     def __init__(self):
#         super(smooth_loss, self).__init__()

#     def forward(self, x):
#         smoothloss = torch.mean(torch.abs(x[:, :, 1:, :]-x[:, :, :-1, :])) + torch.mean(torch.abs(x[:, :, :, 1:]-x[:, :, :, :-1]))
#         return smoothloss

# 总损失:
class total_loss(nn.Module):
    def __init__(self):
        super(total_loss, self).__init__()

    def forward(self, content, content_target, gram_styles, gram_targets, image, α, β):
        closs = content_loss()
        sloss = style_loss()
        smooth = smooth_loss()
        c = closs(content, content_target)
        s = sloss(gram_styles, gram_targets)
        t = α * c + β * s # + γ * smooth(image)
        return t, α * c, β * s




# 只需要卷积层
VGG = vgg19(pretrained=True).features.to(device)

for i, layer in enumerate(VGG):
    # 获取forward过程中网络特定层的输出, 21层用作计算内容损失, 其余用作计算风格损失
    if i in [0,5,10,19,21,28]:
        VGG[i].register_forward_hook(get_features) 
    # 将网络中的最大池化全部替换为平均池化，论文中表示这样效果更好
    elif isinstance(layer, nn.MaxPool2d):
        VGG[i] = nn.AvgPool2d(2)

VGG.eval()
# 由于优化的是生成图，因此冻结网络的参数
for p in VGG.parameters():
    p.requires_grad = False

# 内容损失需要参考的网络输出层
features = []
VGG(content_img.unsqueeze(0))
content_target = features[Content_layer].detach() 
# 风格损失需要参考的网络输出层
features = []
VGG(style_img.unsqueeze(0))
s_targets = features[:4] + features[5:] 
# s_targets = features[:style_layer]
# 计算风格图的格拉姆矩阵:
gram_targets = [gram_matrix(i).detach() for i in s_targets]

# 优化图像就是原图
image = content_img.clone().unsqueeze(0)      
# image = torch.zeros(1,3,512,512).to(device)   # 优化图像是随机噪声

# 牛顿二阶优化法
optimizer = optim.LBFGS([image.requires_grad_()], lr=1.1)  







for step in range(EPOCH):

    features = []
    # LBFGS需要重复多次计算函数，因此需要传入一个闭包去允许它们重新计算你的模型。这个闭包应当清空梯度，计算损失，然后返回
    def closure():
        optimizer.zero_grad()

        VGG(image)
        t_features = features[-6:]
        # 内容层
        content = t_features[Content_layer]  
        # 风格层  
        style_features = t_features[:4] + t_features[5:] 
        # style_features = t_features[:style_layer]
        t_features = []
        # 计算风格层的格拉姆矩阵
        gram_styles = [gram_matrix(i) for i in style_features]  

        # 计算损失
        loss = total_loss()
        tloss, closs[0], sloss[0] = loss(content, content_target, gram_styles, gram_targets, image, α, β)
        tloss.backward()
        return tloss

    optimizer.step(closure)
    
    
    

    

    if step % 2 == 0:
        print('Step {}: Style loss: {:.8f} Content loss: {:.8f}'.format(step, sloss[0], closs[0]))
        temp = decode(image[0].cpu().detach())
        temp = tensor2PIL(temp)
        temp = np.array(temp)
        plt.imsave('result.jpg',temp)
