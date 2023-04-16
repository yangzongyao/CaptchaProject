# %%
from TorchFunction.TorchModel import NeuralNetwork
from TorchFunction.TorchMethod import train_loop, predict
from CaptchaData.CreateCaptchaData_forTorch import captchaData

from CaptchaData.OrcCaptcha import gen_captcha_text_and_image, show_gen_image

import torch
from torch import nn
from torch.utils.data import DataLoader

# %%

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 32
epochs = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

dataset = captchaData(501, device)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

train_loop(dataloader, model, loss_fn, optimizer, epochs=epochs)
# %%
# test code
txt, img = gen_captcha_text_and_image()
predict(model, img, dataset.characters, device)
show_gen_image(txt, img)
# %%
# train data
import numpy as np
idx = np.random.choice(500, 1)[0]
img = dataset[idx][0]
txt_list = dataset[idx][1].argmax(1)
txt = ''.join([dataset.characters[i] for i in txt_list])
show_gen_image(txt, np.array(img).reshape(60,160,3))
x = img.reshape(1,3,60,160)
pred = model(x)
p = pred[0].argmax(1)
''.join([dataset.characters[i] for i in p])
# %%