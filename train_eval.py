from tqdm import tqdm
import torch
import time


def train(model, loader, optimizer, criterion, epochs=20, device="cpu"):
    torch.cuda.reset_max_memory_allocated()

    losses = []
    model.to(device)
    model.train()

    for _ in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            batch_hat = model(batch)
            
            loss = criterion(batch_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() 

        losses.append(total_loss/len(loader))

    torch.save(model, f"{model.__class__.__name__}.pth")
    return losses

def eval_loss(model, loader, criterion, device="cpu"):
    torch.cuda.reset_max_memory_allocated()

    model.eval()
    model.to(device)
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_hat = model(batch)
        loss = criterion(batch_hat, batch)
        total_loss += loss.item() 

    return total_loss/len(loader)

def model_benchmark(model, device="cpu"):
    torch.cuda.reset_max_memory_allocated()

    model.eval()
    model.to(device)
    # Warm-up
    for _ in range(10):
        model(torch.randn(32, 3, 256, 256, device=device))

    # Measure
    start = time.time()
    for _ in range(100):
        _ = model(torch.randn(32, 3, 256, 256, device=device))
    torch.cuda.synchronize()
    latency_ms = (time.time() - start)/100* 1000
    peak_mem_mb = torch.cuda.max_memory_allocated()/(1024**2)
    print(f"Latency: {latency_ms:.2f} ms — Peak Mem: {peak_mem_mb:.1f} MB")