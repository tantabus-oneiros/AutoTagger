import torch
import timm
from timm.models import VisionTransformer
import safetensors.torch
import json

class ImageTagger:
    def __init__(self, model_path, tags_path):
        # Load tags
        with open(tags_path, "r") as file:
            tags = json.load(file)  # type: dict
        self.allowed_tags = list(tags.keys())
        
        for idx, tag in enumerate(self.allowed_tags):
            self.allowed_tags[idx] = tag.replace("_", " ")
        
        # Create model
        self.model = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli",
            pretrained=False,
            num_classes=9083,
        )
        
        # Initialize head
        self.model.head = GatedHead(min(self.model.head.weight.shape), 9083)
        
        # Load model weights
        safetensors.torch.load_model(self.model, model_path)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            if torch.cuda.get_device_capability()[0] >= 7:  # tensor cores
                self.model.to(dtype=torch.float16, memory_format=torch.channels_last)
        
        self.model.eval()
        self.sorted_tag_score = {}
    
    def process_image(self, image, transform, threshold):
        img = image.convert('RGBA')
        tensor = transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            if torch.cuda.get_device_capability()[0] >= 7:  # tensor cores
                tensor = tensor.to(dtype=torch.float16, memory_format=torch.channels_last)
        
        with torch.no_grad():
            probits = self.model(tensor)[0].cpu()
            values, indices = probits.topk(250)
        
        tag_score = dict()
        for i in range(indices.size(0)):
            # Only add tags that are within the range of our allowed_tags list
            if indices[i] < len(self.allowed_tags):
                tag_score[self.allowed_tags[indices[i]]] = values[i].item()
        
        self.sorted_tag_score = dict(sorted(tag_score.items(), key=lambda item: item[1], reverse=True))
        
        return self.create_tags(threshold)
    
    def create_tags(self, threshold):
        filtered_tag_score = {key: value for key, value in self.sorted_tag_score.items() if value > threshold}
        text_no_impl = ", ".join(filtered_tag_score.keys())
        return text_no_impl, filtered_tag_score
    
    def clear(self):
        self.sorted_tag_score = {}
        return "", {}


class GatedHead(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes * 2)
        
        self.act = torch.nn.Sigmoid()
        self.gate = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x[:, :self.num_classes]) * self.gate(x[:, self.num_classes:])
        return x