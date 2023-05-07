import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

def train_epoch(model, data_loader, loss_function, optimizer, scheduler, device, n_accumulated_grads=0):
    model.to(device)
    model.train()
    total_train_loss = 0

    dl_size = len(data_loader)
    
    preds = []
    targets = []

    batch_i = 0
    steps_to_accumulate_grads = 0
    for batch in tqdm(data_loader):
        for key in batch:
            batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()        
        logits = model(batch)
        
        preds.append(logits.argmax(dim=1))
        targets.append(batch['label'])
                
        loss = loss_function(logits, batch['label'])
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if steps_to_accumulate_grads == n_accumulated_grads:
            optimizer.step()
            scheduler.step()
            steps_to_accumulate_grads = 0
        else:
            steps_to_accumulate_grads += 1
            
    if steps_to_accumulate_grads != 0:
        optimizer.step()
        scheduler.step()
    
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = (targets == preds).sum() / preds.shape[0]
    f1 = f1_score(preds.cpu(), targets.cpu())
    
    metrics = {
        "Train Loss": total_train_loss / dl_size,
        "Train Accuracy": acc.item(),
        "Train F1*100": f1.item()*100
    }
    
    
    return metrics
    
    
def eval_epoch(model, data_loader, loss_function, device):
    model.to(device)
    model.eval()
    total_train_loss = 0
    
    preds = []
    targets = []

    dl_size = len(data_loader)

    
    for batch in tqdm(data_loader):
        for key in batch:
            batch[key] = batch[key].to(device)
        
        with torch.no_grad():
            logits = model(batch)
            preds.append(logits.argmax(dim=1))
            targets.append(batch['label'])
        
        loss = loss_function(logits, batch['label'])
        total_train_loss += loss.item()
    
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = (targets == preds).sum() / preds.shape[0]
    f1 = f1_score(preds.cpu(), targets.cpu())
    
    metrics = {
        "Eval Loss": total_train_loss / dl_size,
        "Eval Accuracy": acc.item(),
        "Eval F1": f1.item()*100
    }
    
    return metrics