import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import requests
from io import BytesIO

# ==========================
#      Model Definitions
# ==========================

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features

        self.conv1_1 = vgg_pretrained[0] 
        self.conv1_2 = vgg_pretrained[2] 
        self.pool1   = vgg_pretrained[4]

        self.conv2_1 = vgg_pretrained[5]
        self.conv2_2 = vgg_pretrained[7]
        self.pool2   = vgg_pretrained[9]

        self.conv3_1 = vgg_pretrained[10]
        self.conv3_2 = vgg_pretrained[12]
        self.conv3_3 = vgg_pretrained[14]
        self.conv3_4 = vgg_pretrained[16]
        self.pool3   = vgg_pretrained[18]

        self.conv4_1 = vgg_pretrained[19]
        self.conv4_2 = vgg_pretrained[21]
        self.conv4_3 = vgg_pretrained[23]
        self.conv4_4 = vgg_pretrained[25]
        self.pool4   = vgg_pretrained[27]

        self.conv5_1 = vgg_pretrained[28]
        self.conv5_2 = vgg_pretrained[30]
        self.conv5_3 = vgg_pretrained[32]
        self.conv5_4 = vgg_pretrained[34]
        self.pool5   = vgg_pretrained[36]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1']  = self.pool1(out['r12'])

        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2']  = self.pool2(out['r22'])

        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3']  = self.pool3(out['r34'])

        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4']  = self.pool4(out['r44'])

        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5']  = self.pool5(out['r54'])

        return [out[key] for key in out_keys]

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F_ = input.view(b, c, h * w)
        G = torch.bmm(F_, F_.transpose(1, 2))
        G.div_(h * w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        return nn.MSELoss()(GramMatrix()(input), target)

# ==========================
#   Pre/Post Processing
# ==========================

img_size = 512
img_size_hr = 800

prep = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
])

prep_hr = transforms.Compose([
    transforms.Resize(img_size_hr),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
])

postpa = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1./255)),
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
])
postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    return postpb(t)

def fetch_image_from_url_or_path(source):
    if isinstance(source, str):
        if source.startswith('http://') or source.startswith('https://'):
            response = requests.get(source)
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            return Image.open(source).convert('RGB')
    elif isinstance(source, BytesIO):
        return Image.open(source).convert('RGB')
    else:
        raise ValueError("Unsupported image source format.")

# ==========================
#     Style Transfer Core
# ==========================

def run_style_transfer(content_img, style_img, resolution='low', 
                       max_iter=500, max_iter_hr=200, show_iter=50):
    """
    Args:
        content_img (PIL.Image): Content image (already loaded)
        style_img   (PIL.Image): Style image (already loaded)
        resolution (str): 'low' or 'high' for output quality
        max_iter (int): Iterations for low-res
        max_iter_hr (int): Iterations for high-res
        show_iter (int): Print interval

    Returns:
        out_img_lr (PIL.Image)
        out_img_hr (PIL.Image) or None
    """
    vgg = VGG().cuda() if torch.cuda.is_available() else VGG()

    # Prepare input images
    preprocess = prep
    style_torch   = preprocess(style_img).unsqueeze(0)
    content_torch = preprocess(content_img).unsqueeze(0)

    if torch.cuda.is_available():
        style_torch = style_torch.cuda()
        content_torch = content_torch.cuda()

    style_var = Variable(style_torch)
    content_var = Variable(content_torch)

    style_layers = ['r11','r21','r31','r41','r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    if torch.cuda.is_available():
        loss_fns = [fn.cuda() for fn in loss_fns]

    style_weights = [1e3 / n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    style_targets = [GramMatrix()(A).detach() for A in vgg(style_var, style_layers)]
    content_targets = [A.detach() for A in vgg(content_var, content_layers)]
    targets = style_targets + content_targets

    opt_img = Variable(content_var.data.clone(), requires_grad=True)
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]

    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            loss = sum([weights[i] * loss_fns[i](out[i], targets[i]) for i in range(len(out))])
            loss.backward()
            n_iter[0] += 1
            return loss
        optimizer.step(closure)

    out_img_lr = postp(opt_img.data[0].cpu())

    # If only low-res required
    if resolution == 'low':
        return out_img_lr, None

    # Prepare HR version
    preprocess_hr = prep_hr
    style_torch_hr   = preprocess_hr(style_img).unsqueeze(0)
    content_torch_hr = preprocess_hr(content_img).unsqueeze(0)
    if torch.cuda.is_available():
        style_torch_hr = style_torch_hr.cuda()
        content_torch_hr = content_torch_hr.cuda()

    style_var_hr = Variable(style_torch_hr)
    content_var_hr = Variable(content_torch_hr)

    style_targets_hr = [GramMatrix()(A).detach() for A in vgg(style_var_hr, style_layers)]
    content_targets_hr = [A.detach() for A in vgg(content_var_hr, content_layers)]
    targets_hr = style_targets_hr + content_targets_hr

    opt_img_hr_data = preprocess_hr(out_img_lr).unsqueeze(0)
    if torch.cuda.is_available():
        opt_img_hr_data = opt_img_hr_data.cuda()
    opt_img_hr = Variable(opt_img_hr_data, requires_grad=True)

    optimizer_hr = optim.LBFGS([opt_img_hr])
    n_iter_hr = [0]

    while n_iter_hr[0] <= max_iter_hr:
        def closure_hr():
            optimizer_hr.zero_grad()
            out_hr = vgg(opt_img_hr, loss_layers)
            loss_hr = sum([weights[i] * loss_fns[i](out_hr[i], targets_hr[i]) for i in range(len(out_hr))])
            loss_hr.backward()
            n_iter_hr[0] += 1
            return loss_hr
        optimizer_hr.step(closure_hr)

    out_img_hr = postp(opt_img_hr.data[0].cpu())

    return out_img_lr, out_img_hr
