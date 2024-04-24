import time
from options.train_options import TrainOptions

from data.VCIP_nir2rgb_dataset import *
from models.CycleGanNIR_model import *

from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    dataset1= VCIPNir2RGBDataset_paired(opt) # create dataset
    print("dataset [%s] was created" % type(dataset1).__name__)
    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size,
                    shuffle=not opt.serial_batches, num_workers=int(opt.num_threads))
    dataset_size1 = len(dataset1)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size1)

    dataset2= VCIPNir2RGBDataset(opt) # create dataset
    print("dataset [%s] was created" % type(dataset2).__name__)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size,
                    shuffle=not opt.serial_batches, num_workers=int(opt.num_threads))
    dataset_size2 = len(dataset2)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size2)

    model = CycleGANModel(opt)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        # if epoch > 400 and epoch <= 800:
        #     opt.lambda_A = (800-epoch + 1)/400*10
        #     opt.lambda_B = (800-epoch + 1)/400*30

        if epoch <= 250:
            dataloader = dataloader1
            dataset_size = dataset_size1
            flag = 1
            opt.lr = 0.0001
        # if epoch < 800:
        #     dataloader = dataloader2
        #     dataset_size = dataset_size2
        #     # opt.lambda_A = 10
        #     # opt.lambda_B = 10
        #     flag = 2
        elif epoch%2 == 1:
            dataloader = dataloader2
            dataset_size = dataset_size2
            flag = 2
            opt.lr = 0.00006
        else:
            dataloader = dataloader1
            dataset_size = dataset_size1
            flag = 1
            opt.lr = 0.00006

        for i, data in enumerate(dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(flag)   # calculate loss functions, get gradients, update network weights


            # if total_iters % opt.print_freq == 0:
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # save our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            # model.save_networks('latest')
            model.save_networks(epoch)

        if epoch % opt.save_latest_freq == 0:             # cache our latest model every epoch
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            model.save_networks('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.