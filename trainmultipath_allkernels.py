import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch import nn
import torch 
import os 
import numpy as np
import random
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_everything(123)
    opt = TrainOptions().parse()  
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)  
    print('The number of training images = %d.'% dataset_size)

    model = create_model(opt) #Need to check require grad here
    model.setup(opt)              
    visualizer = Visualizer(opt)   
    total_iters = 0

     #Track losses with wandb
    wandb.init(
        project="MultipathcycleGAN_newkernels",
        config= {
            "batch_size": opt.batch_size,
            "lambda_L2": opt.lambda_L2,
            "lr": opt.lr,
            "n_epochs": opt.n_epochs,
            "n_epochs_decay": opt.n_epochs_decay
        }
    )               

    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    
        if epoch ==1:
            print(f"Current epoch is the first epoch. Therefore, lamdba_L2 is {opt.lambda_L2}")
            opt.lambda_L2 =  opt.lambda_L2
        elif epoch > 1 and epoch <6:
            opt.lambda_L2 =  opt.lambda_L2/100 #Scale lambda in every epoch
            print(f"Scaling down lambda for L2 loss by a factor of 100!.Current value of lambda for L2 is {opt.lambda_L2}")
        else:
            opt.lambda_L2 = 0.01 #fix the lambda for the model for the remaining epochs
            
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0                
        visualizer.reset()              
        model.update_learning_rate()
        for i, data in enumerate(tqdm(dataset)):  
            iter_start_time = time.time()  
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         
            model.optimize_parameters()  

            if total_iters % opt.display_freq == 0:  
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        
