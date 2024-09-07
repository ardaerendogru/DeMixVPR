def custom_scheduler(optimizer, init_lr:float, iter:int, lr_decay_iter:int=1,
                      max_iter:int=45, power:float=0.9):
     
    optimizer.param_groups[0]['lr'] = init_lr*(1 - iter/max_iter)**power