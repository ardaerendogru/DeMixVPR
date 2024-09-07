import torch
import torch.amp as amp
import numpy as np
import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
from prettytable import PrettyTable
import copy
import os
from utils.lr_scheduler import custom_scheduler

def get_validation_recalls(r_list,
                           q_list,
                           q_list_indexes,
                           r_list_indexes, 
                           db_size, 
                           query_size, 
                           k_values, 
                           gt,
                           verbose=False, 
                           faiss_gpu=False, 
                           dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        
        faiss_index.add(r_list)

        
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        
        
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                
                
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if verbose:
            tqdm.write('\n')  
            table = PrettyTable()
            table.field_names = ['K'] + [str(k) for k in k_values]
            table.add_row(['Recall@K'] + [f'{100*v:.2f}' for v in correct_at_k])
            tqdm.write(table.get_string(title=f"Performance on {dataset_name}"))
        
        return d, predictions
    
def train_step(batch, model, loss_fn, miner, optimizer, scaler, multi_model=False):

    if multi_model:
        images, images_depth, labels = batch
    else:
        images, labels = batch
    B, N, ch, h, w = images.shape
    if multi_model:
        images_depth = images_depth.view(-1, ch, h, w).cuda()
    images = images.view(-1, ch, h, w).cuda()
    labels = labels.view(-1).cuda()
    
    acc = 0.0  
    
    
    with amp.autocast(device_type='cuda'):
        if multi_model:
            descriptors = model(images,images_depth)
        else:
            descriptors = model(images)
        if miner is not None:
            miner_outputs = miner(descriptors, labels)
            loss = loss_fn(descriptors, labels, miner_outputs)
            
            
            unique_mined = torch.unique(miner_outputs[0])
            n_mined = unique_mined.numel()
            n_samples = descriptors.size(0)
            acc = 1.0 - (n_mined / n_samples)
        else:
            loss = loss_fn(descriptors, labels)
            if isinstance(loss, tuple):
                loss, acc = loss

    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if multi_model:
        del images, images_depth, labels, descriptors
    else:
        del images, labels, descriptors
    torch.cuda.empty_cache()
    
    return loss, acc


def train_model(**kwargs):
    model = kwargs.get('model')
    loss_fn = kwargs.get('loss_fn')
    miner = kwargs.get('miner')
    optimizer = kwargs.get('optimizer')
    train_loader = kwargs.get('train_dataloader')
    val_loader = kwargs.get('test_dataloader')
    test_dataset = kwargs.get('test_dataset')
    num_epochs = kwargs.get('num_epochs', 30)
    overfitting_detector = kwargs.get('overfitting_detector', False)
    scheduler = kwargs.get('scheduler', None)
    patience = kwargs.get('patience', 5)  
    verbose = kwargs.get('verbose', False)
    train_name = kwargs.get('job_name', 'average_model')
    multi_model = kwargs.get('multi_model', False)
    
    scaler = amp.GradScaler('cuda')  
    db_size = test_dataset._len_db()
    query_size = test_dataset._len_query()
    init_lr = copy.deepcopy(optimizer.param_groups[0]['lr'])
    if overfitting_detector:
        best_r1 = (0, 0)
        best_r5 = (0, 0)
        best_weights = None
        epochs_no_improve = 0  
        
    best_weights = None

    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        
    for epoch in tqdm(range(num_epochs)):
        all_descriptors = torch.tensor([])
        all_indexes = torch.tensor([])
        losses = []
        accs = []
        model.train()
        for batch in train_loader:
            loss, acc = train_step(batch, model, loss_fn, miner, optimizer, scaler, multi_model)
            losses.append(loss.item())
            accs.append(acc)
        
        model.eval()
        with torch.no_grad():  
            with amp.autocast(device_type='cuda'):
                for batch in val_loader:
                    if multi_model:
                        images, images_depth, indexes = batch
                        images_depth = images_depth.cuda()
                    else:
                        images, indexes = batch
                        
                    images = images.cuda()
                    if multi_model:
                        descriptors = model(images, images_depth).cpu()
                    else:
                        descriptors = model(images).cpu()
                    all_descriptors = torch.cat((all_descriptors, descriptors), dim=0)
                    all_indexes = torch.cat((all_indexes, indexes), dim=0)
                    
        database = all_descriptors[query_size:]
        database_indexes = all_indexes[query_size:]
        queries = all_descriptors[:query_size]
        queries_indexes = all_indexes[:query_size]
        recalls_dict, predictions = get_validation_recalls(
            r_list=database, 
            q_list=queries,
            q_list_indexes=queries_indexes,
            r_list_indexes=database_indexes,
            k_values=[1, 5],
            gt=test_dataset.close_indices,
            db_size=db_size,
            query_size=query_size,
            verbose=verbose,
            dataset_name='val_loader',
        )
        if verbose:
            tqdm.write(f'Epoch: {epoch+1}\nR1: {recalls_dict[1]}\nR5: {recalls_dict[5]}\nLoss: {np.mean(losses)}\nAcc: {np.mean(accs)}')
        
        if scheduler == 'custom':
            custom_scheduler(optimizer, init_lr=init_lr, iter=epoch, max_iter=num_epochs, power=0.9)
        elif scheduler == 'cosine':
            scheduler.step()
        
        if overfitting_detector:
            if recalls_dict[1] + recalls_dict[5] > best_r1[1] + best_r5[1]:
                best_r1 = (epoch, recalls_dict[1])
                best_r5 = (epoch, recalls_dict[5])
                torch.save(model.state_dict(), f'./weights/{train_name}.pth')
                epochs_no_improve = 0  
                if verbose:
                    print("Yeyyy! New best model!")
            else:
                epochs_no_improve += 1  
            
            if epochs_no_improve >= patience:
                if verbose:
                    print("Early stopping due to no improvement.")
                model.load_state_dict(torch.load(f'./weights/{train_name}.pth'))
                return model, best_r1, best_r5
        
        # Clear cache at the end of each epoch
        torch.cuda.empty_cache()
    
    return model, best_r1, best_r5
