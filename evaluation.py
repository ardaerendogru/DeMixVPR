import torch
from tqdm import tqdm
import numpy as np
import torch.amp as amp
from prettytable import PrettyTable
import faiss
import faiss.contrib.torch_utils


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
    
def eval_model(**kwargs):
    db_size = kwargs['test_dataset']._len_db()
    query_size = kwargs['test_dataset']._len_query()
    test_dataloader = kwargs['test_dataloader']
    model = kwargs['model']
    verbose = kwargs.get('verbose', True)
    test_dataset = kwargs['test_dataset']
    multi_model = kwargs.get('multi_model', False)
    is_cos_place = kwargs.get('is_cos_place', False)
    
    if is_cos_place and not multi_model:
        model = torch.nn.Sequential(*list(model.children())[:-1])
    all_descriptors = torch.tensor([])
    all_indexes = torch.tensor([])
    with torch.no_grad():  # Disable gradient calculation for validation
        with amp.autocast(device_type='cuda'):
            for batch in test_dataloader:
                if multi_model:
                    images, images_depth, indexes = batch
                    images = images.cuda()
                    images_depth = images_depth.cuda()
                    descriptors = model(images, images_depth).cpu()
                else:
                    images, indexes = batch
                    images = images.cuda()
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
        tqdm.write(f'R1: {recalls_dict[1]}\nR5: {recalls_dict[5]}')
        
    return queries, database, predictions