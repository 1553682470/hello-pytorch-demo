# hello_pytorch.py
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

# 1. 随机生成一张 224×224 的“假超声图”
fake_img = torch.rand(1, 3, 224, 224)

# 2. 用预训练 ResNet18 跑一遍
model = torchvision.models.resnet18(pretrained=True)
model.eval()
with torch.no_grad():
    out = model(fake_img)
pred = torch.argmax(out, 1).item()
print("模型输出类别索引:", pred)

# 3. 保存结果图
plt.imshow(fake_img.squeeze().permute(1, 2, 0))
plt.title("Fake Image")
plt.savefig("result.png")
print("已生成 result.png")
