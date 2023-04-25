# %%
from CaptchaData.CreateCaptchaData_forTorch import captchaData
from torch.utils.data import DataLoader
from CaptchaData.OrcCaptcha import gen_captcha_text_and_image, show_gen_image
import torch
from TorchFunction.TorchModel import NeuralNetwork, ConvolutionalNeuralNetwork, CRNN
from TorchFunction.TorchMethod import train_loop, predict
# %%
device = 'cpu'
batch = 10
dataset = captchaData(20, device, gray=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
model = CRNN()
model.load_state_dict(torch.load('Weights/crnn_100epoch_2000img.pt'))

# %%
# test code
txt, img = gen_captcha_text_and_image()
predict(model, img, dataset.characters, device)
show_gen_image(txt, img)
# %%
# 注意這邊的train set跟訓練用的不會一樣
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
