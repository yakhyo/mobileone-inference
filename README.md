# MobileOne: An Improved One millisecond Mobile Backbone

> [!NOTE]  
> This is only inference code

| Model        | Top-1 Acc. | Latency\* | Model Size (Fused) | Weights(url)                                                                                   |
| ------------ | ---------- | --------- | ------------------ | ---------------------------------------------------------------------------------------------- |
| MobileOne-S0 | 71.4       | 0.79      | 7.96 MB            | [s0_fused](https://github.com/yakhyo/mobileone-inference/releases/download/v0.0.1/s0_fused.pt) |
| MobileOne-S1 | 75.9       | 0.89      | 18.21 MB           | [s1_fused](https://github.com/yakhyo/mobileone-inference/releases/download/v0.0.1/s1_fused.pt) |
| MobileOne-S2 | 77.4       | 1.18      | 29.82 MB           | [s2_fused](https://github.com/yakhyo/mobileone-inference/releases/download/v0.0.1/s2_fused.pt) |
| MobileOne-S3 | 78.1       | 1.53      | 38.48 MB           | [s3_fused](https://github.com/yakhyo/mobileone-inference/releases/download/v0.0.1/s3_fused.pt) |
| MobileOne-S4 | 79.4       | 1.86      | 56.65 MB           | [s4_fused](https://github.com/yakhyo/mobileone-inference/releases/download/v0.0.1/s4_fused.pt) |

\*Latency measured on iPhone 12 Pro.

### Initialize and Re-parameterize

```
from PIL import Image

import torch
from torchvision import transforms
from assets.meta import IMAGENET_CATEGORIES
from models import (
    mobileone_s0,
    mobileone_s1,
    mobileone_s2,
    mobileone_s3,
    mobileone_s4,
    reparameterize_model
)

# Model initialization
model = mobileone_s0(pretrained=True) # it will download unfused weights

# Re-parameterizing stage
model.eval() # change to evaluation mode
reparam_model = reparameterize_model(model)  # re-parameterize model

# Save re-parameterized model `state_dict`
torch.save(reparam_model.state_dict(), "weights/s0_fused.pt") # save re-parameterized model
```

### Model Inference using re-parameterized(fused) model

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobileone_s0(inference_mode=True) # initialize the model in inference mode

# load re-parameterized model weights
model.load_state_dict(torch.load("weights/s0_fused.pt", map_location=device, weights_only=True))
```

Output:

```
INFO: Creating model without pre-trained weights.
<All keys matched successfully>
```

```
# preprocessing input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the model's input size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalize the image using the mean and std of ImageNet
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

def inference(model, image_path):
    model.eval()

    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)

    # Get the top 5 predictions and their confidence scores
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(1)):
        predicted_label = IMAGENET_CATEGORIES[top5_catid[0][i].item()]
        confidence = top5_prob[0][i].item()
        print(f"Predicted class label: {predicted_label}, Confidence: {confidence:.4f}")

inference(model, "assets/tabby_cat.jpg")
```
Input Image:

<img src="assets/tabby_cat.jpg" alt="description" />

Output:

```
Predicted class label: tabby, Confidence: 0.8457
Predicted class label: tiger cat, Confidence: 0.1194
Predicted class label: Egyptian cat, Confidence: 0.0275
Predicted class label: lynx, Confidence: 0.0007
Predicted class label: printer, Confidence: 0.0003
```

### Reference
1. https://github.com/apple/ml-mobileone