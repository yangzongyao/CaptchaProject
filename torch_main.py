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

learning_rate = 1e-4
batch_size = 128
epochs = 500
loss_fn = nn.CrossEntropyLoss()
def word_loss(pred, y):
    loss_of_1_word = loss_fn(pred[:, 0, :], y[:, 0])
    loss_of_2_word = loss_fn(pred[:, 1, :], y[:, 1])
    loss_of_3_word = loss_fn(pred[:, 2, :], y[:, 2])
    loss_of_4_word = loss_fn(pred[:, 3, :], y[:, 3])
    return loss_of_1_word + loss_of_2_word + loss_of_3_word + loss_of_4_word

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = captchaData(501, device)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# %%
# loss test
# x, y = next(iter(dataloader))
# pred = model(x)
# print(pred.argmax(-1)[0])
# print(y[0])
# word_loss(pred, y)

# %%

train_loop(dataloader, model, word_loss, optimizer, epochs=epochs)
# %%
# test code
txt, img = gen_captcha_text_and_image()
predict(model, img, dataset.characters, device)
show_gen_image(txt, img)
# %%
# train data
import numpy as np
idx = np.random.choice(300, 1)[0]
img = dataset[idx][0]
txt = dataset[idx][1].to('cpu')
txt = ''.join(dataset.characters[i] for i in txt)
show_gen_image(txt, np.array(img.to('cpu')).reshape(60,160,3))
x = img.reshape(1,3,60,160)
pred = model(x)
p = pred[0].argmax(1)
''.join([dataset.characters[i] for i in p])
# %%