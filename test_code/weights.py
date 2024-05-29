import torch
import pickle
checkpoint_path = "/data/project/UV-Mamba/result/deform_uvmamba_cityscapes/weights/global_step147312/mp_rank_00_model_states.pt"
pretrained_dict = torch.load(checkpoint_path)["module"]


pretrained_backbone = {k: v for k, v in pretrained_dict.items() if k.startswith("backbone")}
with open("backbone.pkl", "wb") as f:
    pickle.dump(pretrained_backbone, f)

#
# import torch
# import torchvision.models as models
#
# # 加载一个模型，比如 ResNet
# model = models.resnet50(pretrained=True)
#
# # 检查模型的结构
# print(model)
#
# # 假设您知道了 backbone 的层的名称（可以从打印的模型结构中找到）
# backbone_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
#
# # 创建一个字典，用于保存 backbone 的参数
# backbone_params = {}
#
# # 遍历模型的参数
# for name, param in model.named_parameters():
#     # 判断参数名是否在 backbone_layers 中
#     for layer in backbone_layers:
#         if layer in name:
#             # 如果参数名在 backbone_layers 中，将其加入到 backbone_params 中
#             backbone_params[name] = param
#             break
#
# # 创建一个新的 backbone 模型，只包含 backbone 的参数
# backbone_model = torch.nn.Sequential()
#
# # 添加每个层到 backbone_model
# for name, layer in model.named_children():
#     if name in backbone_layers:
#         backbone_model.add_module(name, layer)
#
# # 如果需要，您可以设置 backbone 参数为不需要梯度更新
# for param in backbone_model.parameters():
#     param.requires_grad = False
#
# # 查看 backbone 的参数
# print(backbone_params)
#
# # 查看 backbone 模型
# print(backbone_model)



# model.load_state_dict(pretrained_dict)
# model = model.to(device)
