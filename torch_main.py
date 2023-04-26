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
# device = 'cpu'
print(f"Using {device} device")

# model = NeuralNetwork().to(device)
# model = ConvolutionalNeuralNetwork().to(device)
model = CRNN().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 300
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.CTCLoss()
# def word_loss(pred, y):
#     loss_of_1_word = loss_fn(pred[:, 0, :], y[:, 0])
#     loss_of_2_word = loss_fn(pred[:, 1, :], y[:, 1])
#     loss_of_3_word = loss_fn(pred[:, 2, :], y[:, 2])
#     loss_of_4_word = loss_fn(pred[:, 3, :], y[:, 3])
#     return loss_of_1_word + loss_of_2_word + loss_of_3_word + loss_of_4_word

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = captchaData(20000, device, gray=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# %%

train_loop(dataloader, model, loss_fn, optimizer, epochs=epochs)
torch.save(model.state_dict(), 'crnn_300epoch_20000img.pt')


# %%