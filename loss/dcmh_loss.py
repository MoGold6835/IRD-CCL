import torch

def DCMH_loss(B, F, G, Sim, gamma, eta):
    theta = 1.0 / 2 * torch.mm(G, F.t())
    term1 = torch.sum(torch.log(1.0 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow((B - F), 2) + torch.pow((B - G), 2))
    term3 = torch.sum(torch.pow(torch.mm(F, torch.ones((F.shape[1], 1), device=F.device)), 2)) + torch.sum(
        torch.pow(torch.mm(G, torch.ones((G.shape[1], 1), device=G.device)), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss