import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict
from torch.autograd import Variable
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
mainpath = os.getcwd()
pix2pixhd_dir = Path(mainpath+'/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

# src 폴더 내에 pix2pixHD 파일로 경로설정

from data.data_loader import CreateDataLoader
# CreateDataLoader(opt) 함수를 사용해서 dataloadaer 생성
from models.models import create_model
# create_model(opt) 함수를 불러와서 해당 모델 사용
import util.util as util
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
from util.visualizer import Visualizer
# 현재 훈련되고 있는 상황 그래프, 수치로 나타내기
import src.config.train_opt as opt
# 여기에서 쓰이는 option들
'''
batchSize=1
beta1=0.5
checkpoints_dir='./checkpoints/'
continue_train=False
data_type=32
dataroot='./data/2/train'
# load_pretrain = './checkpoints/target/' # use this if you want to continue last training
debug=False
display_freq=640
display_winsize=512
feat_num=3
fineSize=512
fine_size=480
input_nc=3
instance_feat=False
isTrain=True
label_feat=False
label_nc=36
lambda_feat=10.0
loadSize=512
load_features=False
load_pretrain=''
lr=0.0002
max_dataset_size=100000
model='pix2pixHD'
nThreads=2
n_blocks_global=9
n_blocks_local=3
n_clusters=10
n_downsample_E=4
n_downsample_global=4
n_layers_D=3
n_local_enhancers=1
name='2_without'
ndf=64
nef=16
netG='global'
ngf=64
niter=20
niter_decay=20
niter_fix_global=0
no_flip=False
no_ganFeat_loss=False
no_html=False
no_instance=True
no_lsgan=False
no_vgg_loss=False
norm='instance'
num_D=2
output_nc=3
phase='train'
pool_size=0
print_freq=640
resize_or_crop='scale_width'
save_epoch_freq=10
save_latest_freq=640
serial_batches=False
tf_log=True
use_dropout=False
verbose=False
which_epoch='latest'
gpu_ids=[0,1,2,3]
'''
# pix2pixHD 폴더 내에 여러가지 모듈들을 import 해온다.
# pix2pixHD 모델 사용

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

# 기본 pytorch multiprocessing file system 설정

def main(): # 입력 X, return X
	iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
	# 반복 경로 받아오기
	data_loader = CreateDataLoader(opt)
	# option에 해당하는 data_loader 생성

	dataset = data_loader.load_data()
	# dataset을 data_loader로부터 받아온다.
	dataset_size = len(data_loader)
	# dataset의 사이즈를 지정
	print('#training images = %d' % dataset_size)

	start_epoch, epoch_iter = 1, 0
	total_steps = (start_epoch - 1) * dataset_size + epoch_iter
	display_delta = total_steps % opt.display_freq
	print_delta = total_steps % opt.print_freq
	save_delta = total_steps % opt.save_latest_freq
	# delta 값들 지정

	model = create_model(opt)
	# model = model.cuda()
	visualizer = Visualizer(opt)
	# 현재 option에 해당하는 훈련 과정 출력

	for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
		# 총 40번 반복
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
