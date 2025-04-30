import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# Model Architecture from the reference notebook
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv2d(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return self.conv2d(self.reflection_pad(x))


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        return self.model(x)


def load_image(image, size=None, scale=None):
    """
    Process image exactly like in the Colab notebook
    
    Args:
        image (PIL.Image): The image to process
        size (int, optional): Size to resize the image to (square)
        scale (float, optional): Scale factor to resize by
        
    Returns:
        PIL.Image: Processed image
    """
    # Convert to RGB if not already
    img = image.convert('RGB')
    
    if size is not None:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    elif scale is not None:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    # else: keep original size
    return img


def run_fast_style_transfer(model_path, content_image_pil, use_full_size=True, resize_dim=512):
    """
    Applies fast style transfer using a pre-trained TransformerNet model.
    Implementation exactly matches the Colab notebook.

    Args:
        model_path (str): Path to the pre-trained .pth model file.
        content_image_pil (PIL.Image): The content image as a PIL object.
        use_full_size (bool): If True, process the image at its original size. 
                             If False, resize to resize_dim.
        resize_dim (int): The dimension to resize to if use_full_size is False.

    Returns:
        PIL.Image: The stylized image as a PIL object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fast Transfer - Using device: {device}")

    # Process image using the exact same approach as Colab
    if not use_full_size:
        content_image = load_image(content_image_pil, size=resize_dim)
        print(f"Resizing image to {resize_dim}x{resize_dim}")
    else:
        content_image = load_image(content_image_pil)
        print(f"Using full image size: {content_image.size}")

    # Load model - exactly like Colab
    model = TransformerNet().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Clean deprecated keys - exactly like Colab
        for k in list(state_dict.keys()):
            if "running_mean" in k or "running_var" in k:
                del state_dict[k]
                
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded fast transfer model from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load model state_dict: {e}")
        raise

    # Transform image - exactly like Colab notebook
    transform = transforms.ToTensor()
    content_tensor = transform(content_image).unsqueeze(0).mul(255).to(device)
    print(f"Input tensor shape: {content_tensor.shape}, "
          f"range: ({content_tensor.min().item():.1f}, {content_tensor.max().item():.1f})")

    # Stylize
    with torch.no_grad():
        output_tensor = model(content_tensor).cpu()

    # Convert tensor back to image - exactly like Colab
    output_image = output_tensor.squeeze(0).clamp(0, 255).numpy()
    output_image = output_image.transpose(1, 2, 0).astype('uint8')
    output_image_pil = Image.fromarray(output_image)
    
    print(f"Fast style transfer complete. Output size: {output_image_pil.size}")
    return output_image_pil


# Example usage (for testing this file directly)
if __name__ == '__main__':
    # Create a dummy white image for testing
    dummy_image = Image.new('RGB', (600, 400), color='white')
    # Make sure the model path is correct for your setup
    # This path assumes the script is run from the root directory
    test_model_path = '../models/starry_night.pth'
    
    print("Testing run_fast_style_transfer...")
    try:
        stylized_img_full = run_fast_style_transfer(test_model_path, dummy_image, use_full_size=True)
        stylized_img_resized = run_fast_style_transfer(test_model_path, dummy_image, use_full_size=False, resize_dim=256)
        
        print(f"Full size output: {stylized_img_full.size}")
        print(f"Resized output: {stylized_img_resized.size}")
        
        # Optionally save the test images
        # stylized_img_full.save("test_stylized_full.jpg")
        # stylized_img_resized.save("test_stylized_resized.jpg")
        print("Test successful.")
        
    except FileNotFoundError:
        print(f"ERROR: Test failed. Model not found at {test_model_path}. Ensure the path is correct and the file exists.")
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}") 