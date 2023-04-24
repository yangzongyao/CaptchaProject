# %%
from TorchFunction.TorchModel import NeuralNetwork, ConvolutionalNeuralNetwork, CRNN
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
device = 'cpu'
print(f"Using {device} device")

# model = NeuralNetwork().to(device)
# model = ConvolutionalNeuralNetwork().to(device)
model = CRNN().to(device)

learning_rate = 1e-3
batch_size = 32
epochs = 100
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.CTCLoss()
def word_loss(pred, y):
    loss_of_1_word = loss_fn(pred[:, 0, :], y[:, 0])
    loss_of_2_word = loss_fn(pred[:, 1, :], y[:, 1])
    loss_of_3_word = loss_fn(pred[:, 2, :], y[:, 2])
    loss_of_4_word = loss_fn(pred[:, 3, :], y[:, 3])
    return loss_of_1_word + loss_of_2_word + loss_of_3_word + loss_of_4_word

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = captchaData(20, device, gray=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# %%

train_loop(dataloader, model, loss_fn, optimizer, epochs=epochs)
# torch.save(model.state_dict(), 'cnn_500epoch_20000img.pt')

# %%
# test code
txt, img = gen_captcha_text_and_image()
predict(model, img, dataset.characters, device)
show_gen_image(txt, img)
# %%
# train data
import numpy as np
idx = np.random.choice(dataset.n_samples, 1)[0]
img = dataset[idx][0]
txt = dataset[idx][1].to('cpu')
txt = ''.join(dataset.characters[i] for i in txt)
show_gen_image(txt, np.array(img.to('cpu')).reshape(60,160,1))
X = img.reshape(1,1,60,160)
X = X / 255
pred = model(X)
p = pred[:,0,:].argmax(1)
''.join([dataset.characters[i] for i in p])

# %%
x, y = next(iter(dataloader))
x = x / 255
pred = model(x)
# pred = nn.functional.log_softmax(pred, dim=2)
pred = pred.log_softmax(2).detach()
pred_space_length = pred.shape[0]
y_space_length = y.shape[1]
# loss = loss_fn(pred, y)
input_lengths = torch.full(size=(x.shape[0],), fill_value=pred_space_length, dtype=torch.long)
target_lengths = torch.full(size=(x.shape[0],), fill_value=y_space_length, dtype=torch.long)

loss = loss_fn(log_probs=pred, targets=y, target_lengths=target_lengths, input_lengths=input_lengths)
print(pred[:,0,:].argmax(1))
print(y[0,:])
print(loss)
# %%
