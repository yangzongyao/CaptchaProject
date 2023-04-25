# %%
import torch
import numpy as np

# %%
def train_loop(dataloader, model, loss_fn, optimizer, epochs):
    epoch_size = epochs
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            # 測試CRNN
            batch_size = X.shape[0]
            X = X / 255
            pred = model(X)
            pred = pred.log_softmax(2)
            pred_space_length = pred.shape[0]
            y_space_length = y.shape[1]
            # loss = loss_fn(pred, y)
            input_lengths = torch.full(size=(batch_size,), fill_value=pred_space_length, dtype=torch.long)
            target_lengths = torch.full(size=(batch_size,), fill_value=y_space_length, dtype=torch.long)

            loss = loss_fn(log_probs=pred, targets=y, target_lengths=target_lengths, input_lengths=input_lengths)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 30 == 0:
                pass
                # loss, current = loss.item(), (batch + 1) * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        pred = model(X)
        pred = pred.log_softmax(2)
        loss = loss_fn(log_probs=pred, targets=y, target_lengths=target_lengths, input_lengths=input_lengths)

        # loss = loss_fn(pred, y)
        loss, current = loss.item(), (epoch + 1)
        print(f"epoch loss: {loss:>7f}  [{current:>5d}/{epoch_size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def predict(model, X, characters, device):
    model.eval()
    X = np.dot(X, [0.2989, 0.5870, 0.1140])
    X = X.reshape(1,1,60,160).astype(np.float32)
    X = X / 255
    pred = model(torch.from_numpy(X).to(device))
    text_idx = pred[:,0,:].argmax(1)
    print(''.join([characters[i] for i in text_idx]))
# %%
