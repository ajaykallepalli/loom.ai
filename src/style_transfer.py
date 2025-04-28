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
        # Load VGG19 model pre-trained on ImageNet
        # We only need the features part, not the classifier
        vgg_pretrained = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Extract specific layers needed for style/content loss
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

        # Freeze VGG parameters - we don't train VGG
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, out_keys):
        """Forward pass through VGG, returning activations from specified layers."""
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
    """Computes the Gram matrix for style loss."""
    def forward(self, input):
        b, c, h, w = input.size() # batch, channels, height, width
        F_ = input.view(b, c, h * w) # Reshape features
        # Compute Gram matrix (matrix multiply features by their transpose)
        G = torch.bmm(F_, F_.transpose(1, 2))
        # Normalize by the number of elements (h*w)
        G.div_(h * w)
        return G

class GramMSELoss(nn.Module):
    """Computes MSE loss between input Gram matrix and target Gram matrix."""
    def forward(self, input, target):
        return nn.MSELoss()(GramMatrix()(input), target)

# ==========================
#   Pre/Post Processing
# ==========================

# Image size for processing
img_size = 512
img_size_hr = 800 # For optional high-resolution pass

# VGG uses specific normalization constants (BGR format, range 0-255)
# Reference: https://pytorch.org/vision/stable/models.html
# Note: The original reference code used slightly different values and
# inverted channels (Lambda lambda x: x[torch.LongTensor([2,1,0])]).
# Using standard ImageNet normalization and letting Torch handle RGB/BGR.
prep = transforms.Compose([
    transforms.Resize((img_size, img_size)), # Ensure square resize
    transforms.ToTensor(), # Converts PIL image (0-255) to Tensor (0.0-1.0)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

prep_hr = transforms.Compose([
    transforms.Resize((img_size_hr, img_size_hr)), # Ensure square resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to convert a tensor back to a PIL image
def tensor_to_pil(tensor):
    # Clone tensor to avoid modifying the original
    image = tensor.cpu().clone().squeeze(0) # Remove batch dimension
    # Denormalize: multiply by std deviation, add mean
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = inv_normalize(image)
    # Clamp values to [0, 1] range after denormalization
    image = torch.clamp(image, 0, 1)
    # Convert tensor to PIL image
    image = transforms.ToPILImage()(image)
    return image

# This function isn't used by the current inference.py, which gets PIL images directly
# Keeping it here for potential standalone use or future reference.
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
                       max_iter=300, max_iter_hr=200, show_iter=50):
    """
    Performs Neural Style Transfer using the LBFGS optimization method.

    Args:
        content_img (PIL.Image): Content image (already loaded).
        style_img   (PIL.Image): Style image (already loaded).
        resolution (str): 'low' (default, 512px) or 'high' (800px) for output quality.
        max_iter (int): Max optimization iterations for low-res pass.
        max_iter_hr (int): Max optimization iterations for high-res pass.
        show_iter (int): Print interval (not used in server context).

    Returns:
        tuple: (out_img_lr, out_img_hr)
               out_img_lr (PIL.Image): Stylized low-resolution image.
               out_img_hr (PIL.Image or None): Stylized high-resolution image (if resolution='high'), else None.
    """
    print(f"Running style transfer with resolution: {resolution}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vgg = VGG().to(device) # Load VGG onto the correct device

    # Prepare input images
    preprocess = prep
    # Process images and move tensors to the correct device
    style_torch   = preprocess(style_img).unsqueeze(0).to(device)
    content_torch = preprocess(content_img).unsqueeze(0).to(device)

    # Create torch Variables (legacy, but used in original code)
    # For newer PyTorch, using tensors directly is fine if requires_grad=True is set later
    style_var = Variable(style_torch)
    content_var = Variable(content_torch)

    # Layers for style and content loss calculation
    style_layers = ['r11','r21','r31','r41','r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    
    # Loss functions (GramMSELoss for style, MSELoss for content)
    loss_fns = [GramMSELoss().to(device)] * len(style_layers) + [nn.MSELoss().to(device)] * len(content_layers)

    # Weights for style and content losses (tunable hyperparameters)
    style_weights = [1e3 / n**2 for n in [64,128,256,512,512]] # Original weights
    content_weights = [1e0] # Original weight
    weights = style_weights + content_weights

    # Compute target features/Gram matrices (once per image)
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_var, style_layers)]
    content_targets = [A.detach() for A in vgg(content_var, content_layers)]
    targets = style_targets + content_targets

    # Initialize the output image as a clone of the content image
    # Set requires_grad=True to enable optimization
    opt_img = Variable(content_var.data.clone(), requires_grad=True)
    
    # Optimizer: LBFGS is effective for style transfer but can be slow
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0] # Using list to allow modification inside closure

    print(f"Starting optimization (max_iter={max_iter})...")
    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            # Calculate weighted loss
            layer_losses = [weights[i] * loss_fns[i](out[i], targets[i]) for i in range(len(loss_layers))]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            if n_iter[0] % show_iter == 0:
                 print(f'Iteration {n_iter[0]}, Loss: {loss.item():.4f}')
            return loss
        optimizer.step(closure)
    print(f"Optimization finished after {n_iter[0]-1} iterations.")

    # Convert the optimized tensor back to a PIL image
    out_img_lr = tensor_to_pil(opt_img.data)

    # If only low-res required, return now
    if resolution == 'low':
        print("Low resolution pass complete.")
        return out_img_lr, None

    # --- Optional High-Resolution Pass ---
    print("Starting high-resolution pass...")
    preprocess_hr = prep_hr
    style_torch_hr   = preprocess_hr(style_img).unsqueeze(0).to(device)
    content_torch_hr = preprocess_hr(content_img).unsqueeze(0).to(device)

    style_var_hr = Variable(style_torch_hr)
    content_var_hr = Variable(content_torch_hr)

    # Compute HR targets
    style_targets_hr = [GramMatrix()(A).detach() for A in vgg(style_var_hr, style_layers)]
    content_targets_hr = [A.detach() for A in vgg(content_var_hr, content_layers)]
    targets_hr = style_targets_hr + content_targets_hr

    # Initialize HR output image using the low-res result
    opt_img_hr_data = preprocess_hr(out_img_lr).unsqueeze(0).to(device)
    opt_img_hr = Variable(opt_img_hr_data, requires_grad=True)

    optimizer_hr = optim.LBFGS([opt_img_hr])
    n_iter_hr = [0]

    print(f"Starting HR optimization (max_iter_hr={max_iter_hr})...")
    while n_iter_hr[0] <= max_iter_hr:
        def closure_hr():
            optimizer_hr.zero_grad()
            out_hr = vgg(opt_img_hr, loss_layers)
            layer_losses_hr = [weights[i] * loss_fns[i](out_hr[i], targets_hr[i]) for i in range(len(loss_layers))]
            loss_hr = sum(layer_losses_hr)
            loss_hr.backward()
            n_iter_hr[0] += 1
            if n_iter_hr[0] % show_iter == 0:
                print(f'HR Iteration {n_iter_hr[0]}, Loss: {loss_hr.item():.4f}')
            return loss_hr
        optimizer_hr.step(closure_hr)
    print(f"HR optimization finished after {n_iter_hr[0]-1} iterations.")

    out_img_hr = tensor_to_pil(opt_img_hr.data)
    print("High resolution pass complete.")

    return out_img_lr, out_img_hr

# Example Usage (optional, for local testing)
if __name__ == '__main__':
    # Create dummy images or load from files for testing
    dummy_content = Image.new('RGB', (256, 256), color = 'red')
    dummy_style = Image.new('RGB', (256, 256), color = 'blue')
    
    # You would load your actual model here before calling stylize_image
    # model = load_my_model()
    # device = torch.device(...)

    # For placeholder, we pass None
    result_image = run_style_transfer(dummy_content, dummy_style) 
    if result_image:
        result_image.save("stylized_placeholder.jpg")
        print("Saved stylized_placeholder.jpg") 