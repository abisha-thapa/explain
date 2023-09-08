import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
num_epochs = 1000
hidden_dim = 16

def train(model_object, G, hidden_dim, num_classes, graph_num=''):
    model = model_object(G.ndata['feat'].shape[1], hidden_dim, num_classes).to(device)
    G = G.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        logits = model(G, G.ndata['feat'])
        loss = F.cross_entropy(logits[G.ndata['train_mask']], G.ndata['label'][G.ndata['train_mask']])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate accuracy on validation set
        model.eval()
        with torch.no_grad():
            logits = model(G, G.ndata['feat'])
            pred = logits.argmax(dim=1)
            acc = (pred[G.ndata['val_mask']] == G.ndata['label'][G.ndata['val_mask']]).float().mean()

        if epoch%100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Val Acc: {acc.item()}\n")

    # Evaluate accuracy on test set
    model.eval()
    with torch.no_grad():
        logits = model(G, G.ndata['feat'])
        pred = logits.argmax(dim=1)
        test_acc = (pred[G.ndata['test_mask']] == G.ndata['label'][G.ndata['test_mask']]).float().mean()

    print(f" Test Acc: {test_acc.item()}")

    return model, pred