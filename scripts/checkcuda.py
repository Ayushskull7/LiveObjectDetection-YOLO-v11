import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.__version__)
print(torch.version.cuda)
device="cuda"
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = x + y
print(z.device)