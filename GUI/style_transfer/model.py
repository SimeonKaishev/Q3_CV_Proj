import torch
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights

class StyleTransferModel:
    def __init__(self, content_img, style_img):
        # 🔹 Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # ✅ Debugging GPU vs. CPU usage

        # Load VGG19 model on GPU
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()

        # Define layers
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        self.content_layers = ["conv4_1"]

        # Move images to GPU
        self.content_img = content_img.to(self.device)
        self.style_img = style_img.to(self.device)

    def get_features(self, image):
        """Extracts feature maps from layers in VGG19."""
        layers = {
            "conv1_1": 0, "conv2_1": 5, "conv3_1": 10,
            "conv4_1": 19, "conv4_2": 21, "conv5_1": 28
        }

        features = {}
        x = image.to(self.device)  # Move input image to GPU

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in layers.values():
                features[list(layers.keys())[list(layers.values()).index(i)]] = x

        return features

    def gram_matrix(self, tensor):
        """Computes the Gram Matrix for style representation."""
        _, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (c * h * w)

    def train(self, num_steps=500, style_weight=1e6, content_weight=1):
        """Optimizes the input image to apply style transfer using GPU (if available)."""
        input_img = self.content_img.clone().requires_grad_(True).to(self.device)
        optimizer = torch.optim.Adam([input_img], lr=0.003)
        loss_fn = torch.nn.MSELoss()

        for i in range(num_steps):
            optimizer.zero_grad()
            content_features = self.get_features(self.content_img)
            style_features = self.get_features(self.style_img)
            input_features = self.get_features(input_img)

            content_loss = loss_fn(input_features["conv4_1"], content_features["conv4_1"]) * content_weight

            style_loss = sum(
                loss_fn(self.gram_matrix(input_features[layer]), self.gram_matrix(style_features[layer]))
                for layer in self.style_layers
            ) / len(self.style_layers)

            total_loss = content_loss + (style_weight * style_loss)
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Step {i}/{num_steps} - Loss: {total_loss.item()} (Running on {self.device})")

        return torch.clamp(input_img.detach().cpu(), 0, 1)  # ✅ Fix: Ensure values are in range [0,1]
