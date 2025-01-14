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
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import wandb



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
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # Creates a dataset with all the data in the opt.dataroot directory.
    dataset_size = len(dataset)    # Total length of datatset. Returns maximum of all possible datasets.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    #Track losses with wandb
    wandb.init(
        project="MultipathCycleGAN_with_Anatomycontext_4domains",
        config= {
            "batch_size": opt.batch_size,
            "lambda_L2": opt.lambda_L2,
            "lambda_seg": opt.lambda_seg,
            "lr": opt.lr,
            "n_epochs": opt.n_epochs,
            "n_epochs_decay": opt.n_epochs_decay
        }
    )

    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        """For a batch size of 1, we see that the L2 loss forces the UNet to converge to identity in an epoch.
        From the second epoch onwards, the L2 should be weighted lesser and lesser to let the adversarial take over. 
        The lambda value for L2 should be scaled down by a factor of 100 from epoch 2."""
        if epoch == 1 and opt.continue_train == False:
            print(f"Current epoch is the first epoch. Therefore, lamdba_L2 is {opt.lambda_L2}")
            opt.lambda_L2 =  opt.lambda_L2
        elif epoch > 1 and epoch <6 and opt.continue_train == False:
            opt.lambda_L2 =  opt.lambda_L2/100
            print(f"Scaling down lambda for L2 loss by a factor of 100!.Current value of lambda for L2 is {opt.lambda_L2}")
        else:
            opt.lambda_L2 = 0.01 #fix the lambda for the model for the remaining epochs

        # indices = torch.randperm(fract_dataset)
        # subset_dataset = SubsetRandomSampler(dataset, indices) 
        # random_loader = DataLoader(subset_dataset, batch_size=opt.batch_size, sampler=subset_dataset, num_workers=opt.num_threads)    
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(tqdm(dataset)):

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                wandb.log({k: v for k, v in losses.items()}) #Log losses onto weights and biases
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    wandb.finish()

        
