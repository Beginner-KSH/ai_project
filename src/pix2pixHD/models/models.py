### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

# 각 option에 맞는 모델을 불러온다

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()        
    elif opt.model == 'pix2pixHD_Temporal':
        from .pix2pixHD_model_temperal import Pix2PixHDModel_Temperal, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel_Temperal()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_face_gan':
        from .pix2pixHD_face_gan import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    # if opt.isTrain and len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
