import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict
# 입력순서를 기억하고 있는 딕셔너리
from torch.autograd import Variable
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
mainpath = os.getcwd()
pix2pixhd_dir = Path(mainpath+'/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))


from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import src.config.train_opt_temperal as opt

# 여기에서 쓰이는 option들
'''
aspect_ratio=1.0
batchSize=1
checkpoints_dir='./checkpoints/'
cluster_path='features_clustered_010.npy'
data_type=32
dataroot='./data/1/test'
display_winsize=512
engine=None
export_onnx=None
feat_num=3
fineSize=512
fine_size=480
how_many=50
input_nc=3
instance_feat=False
isTrain=False
label_feat=False
label_nc=36
loadSize=512
load_features=False
max_dataset_size=100000
model='pix2pixHD_Temporal'
nThreads=1
n_blocks_global=9
n_blocks_local=3
n_clusters=10
n_downsample_E=4
n_downsample_global=4
n_local_enhancers=1
name='updated'
nef=16
netG='global'
ngf=64
niter_fix_global=0
no_flip=True
no_instance=True
norm='instance'
ntest=100000
onnx=None
output_nc=3
phase='test'
resize_or_crop='scale_width'
results_dir='./results/'
serial_batches=False
tf_log=False
use_dropout=False
verbose=False
which_epoch='latest'
gpu_ids = [0]
is_temporal=True
'''

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True


def main():
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    model = create_model(opt)
    model = model.cuda()
    visualizer = Visualizer(opt)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']), infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ############### Backward Pass ####################
            # update generator weights
            model.optimizer_G.zero_grad()
            loss_G.backward()
            model.optimizer_G.step()

            # update discriminator weights
            model.optimizer_D.zero_grad()
            loss_D.backward()
            model.optimizer_D.step()


            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
                
        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
