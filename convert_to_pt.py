import torch

model = torch.load('/Users/borosabel/Doc/Egyetem/Signlanguagerecognition/PyCharm-mediapipe/model.pth', map_location=torch.device('cpu'))
model = model.eval()
input_tensor = torch.rand(1, 3, 224, 224)
script_model = torch.jit.trace(model, input_tensor)
script_model.save('/Users/borosabel/Doc/Egyetem/Signlanguagerecognition/PyCharm-mediapipe/SavedModels/model.pt')
