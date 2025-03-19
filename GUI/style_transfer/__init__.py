import torch
from torchvision.models import vgg19, VGG19_Weights

class StyleTransferModel:
    def __init__(self, content_img, style_img):
        # ✅ Force GPU usage if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Debugging info

        # ✅ Move model to GPU
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()

        # Define layers
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        self.content_layers = ["conv4_1"]

        # ✅ Move images to GPU
        self.content_img = content_img.to(self.device)
        self.style_img = style_img.to(self.device)
