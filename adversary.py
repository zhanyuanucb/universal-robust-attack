import torch.nn as nn
import numpy as np
from transforms import*
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

def eot_craft(model: nn.Module, T: list, random_transform: RandomTransform, 
              adv_img: np.array, adv_label: int, 
              steps: int, step_size: float, targeted: bool, 
              eps_below: np.array, eps_above: np.array, device,
              clip_below=0., clip_above=1., criterion=None, 
              verbose=True, log_interval=100):
    
    def _craft_step(adv_img):
        average_loss, N = 0., len(T)
        for i, t in enumerate(T):
            x_t = torch.from_numpy(t(adv_img)).float().to(device)
            x_t.requires_grad_(True)
            logits = model(x_t)
            loss = criterion(logits, adv_label)/N
            loss.backward()
            if targeted:
                adv_img -= step_size*np.sign(x_t.grad[0].cpu().numpy())
            else:
                adv_img += step_size*np.sign(x_t.grad[0].cpu().numpy())
            adv_img = np.clip(np.clip(adv_img, eps_below, eps_above), clip_below, clip_above)

            average_loss += loss.item()
        average_loss /= N
        return adv_img, average_loss
    
    adv_label = torch.Tensor([adv_label]).to(torch.int64).to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    for step in range(1, steps+1):
        adv_img, average_loss = _craft_step(adv_img)
        if verbose and step % log_interval == 0:
            with torch.no_grad():
                x_t = random_transform(adv_img)
                logits = model(torch.from_numpy(x_t).float().to(device))
            logits = logits.cpu().numpy()
            pred = np.argmax(logits)
            plt.imshow(x_t[0][0], cmap="gray")
            plt.title(f"Step {step}: Prediction: {pred} \n Average_loss = {average_loss:.3f}")
            plt.show()
    return adv_img

def adv_eval(model, eval_steps, adv_img, 
             ori_label, adv_label, targeted, 
             random_transform, device, verbose=True):
    fail_count = succ_count = 0
    other_count = 0 if targeted else None
    for _ in range(eval_steps):
        with torch.no_grad():
            x_t = random_transform(adv_img)
            logits = model(torch.from_numpy(x_t).float().to(device))
            logits = logits.cpu().numpy()
            pred = np.argmax(logits)
        if targeted:
            if pred == adv_label:
                succ_count += 1
            elif pred == ori_label:
                fail_count += 1
            else:
                other_count += 1
        else:
            if pred != ori_label:
                succ_count += 1
            else:
                fail_count += 1
    succ_rate = succ_count/eval_steps
    fail_rate = fail_count/eval_steps
    if other_count is not None:
        other_rate = other_count/eval_steps
        if verbose:
            print(f"Success rate: {succ_rate:.3f}; Failure rate: {fail_rate:.3f}; Others: {other_rate:.3f}")
        return succ_rate, fail_rate, other_rate
    else:
        if verbose:
            print(f"Success rate: {succ_count/eval_steps:.3f}; Failure rate: {fail_count/eval_steps:.3f}")
        return succ_rate, fail_rate
      

# Universal&Robust pertubation crafting via EoT
def uni_eot_craft(model: nn.Module, T: list,
                  dataloader: DataLoader, epochs: int, step_size: float,
                  eps: float, device, shape=(1, 1, 28, 28), optimizer=None, 
                  clip_below=0., clip_above=1.,
                  criterion=None, verbose=True, log_interval=100):
    
    def _uni_craft_step(uni_pert):
        max_loss, average_loss, N = 0., 0., len(T)
        for i, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            for j, t in enumerate(T):
                uni_pert_nxt = Variable(uni_pert, requires_grad=True)
                x_adv = x + uni_pert_nxt
                x_adv = torch.clamp(x_adv, clip_below, clip_above)
                x_t = t(x_adv)
                logits = model(x_t)
                loss = criterion(logits, label)/N
                loss.backward()
                
                # Update the perturbation
                if optimizer is not None:
                    uni_pert = uni_pert - optimizer(uni_pert_nxt.grad)
                else:
                    uni_pert = uni_pert + step_size*uni_pert_nxt.grad
                average_loss += loss.item()
                max_loss = max(max_loss, loss.item())
                
            uni_pert = torch.clamp(uni_pert, -eps, eps)
        average_loss /= (i+1)*N
        return uni_pert, average_loss, max_loss
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    uni_pert = torch.zeros(shape).to(device)
    for epoch in range(1, epochs+1):
        uni_pert, average_loss, max_loss = _uni_craft_step(uni_pert)
        if verbose and epoch % log_interval == 0:
            plt.imshow(uni_pert.detach().cpu().numpy()[0][0], cmap="gray")
            plt.title(f"Epoch {epoch}: \n Average loss = {average_loss:.3f}; Max loss = {max_loss:.3f}")
            plt.axis("off")
            plt.show()
    return uni_pert

# Evaluation function
def uni_adv_eval(model, uni_pert, random_transform,
                 testloader, device, verbose=True, log_interval=100):
    succ, total = 0, 0
    for i, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        total += images.size(0)
        x = torch.clamp(images+uni_pert, 0., 1.)
        x_t = random_transform(x)
        
        with torch.no_grad():
            logits = model(x_t)
        _, predicted = logits.max(1)
        succ += predicted.ne(labels).sum().item()
        
        if verbose and total % log_interval == 0:
            x_np = x[0][0].cpu().numpy()
            xt_np = x_t[0][0].cpu().numpy()
            x_diff = xt_np - x_np
            fig = plt.figure(figsize=(8, 4))
            ax = plt.subplot(131)
            ax.set_title(f'adv \n Success rate: {succ/total:.3f}')
            plt.imshow(x_np, cmap="gray")
            plt.axis("off")
            ax = plt.subplot(132)
            ax.set_title('t(adv)')
            plt.imshow(xt_np, cmap="gray")
            plt.axis("off")
            ax = plt.subplot(133)
            ax.set_title("t(adv) - adv")
            plt.imshow(x_diff, cmap="gray")
            plt.axis("off")
            plt.show()
            
    succ_rate = succ/total
    print(f"Success rate: {succ_rate}")
    return succ_rate