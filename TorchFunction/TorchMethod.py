# %%
import torch

# %%
def train_loop(dataloader, model, loss_fn, optimizer, epochs):
    epoch_size = epochs
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X / 255
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        pred = model(X)
        loss = loss_fn(pred, y)
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
    X = X.reshape(1,3,60,160)
    X = X / 255
    pred = model(torch.from_numpy(X).to(device))
    text_idx = pred[0].argmax(1)
    print(''.join([characters[i] for i in text_idx]))
# %%
