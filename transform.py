import torch
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF


class Transform:
    def __init__(self, flip=True, r_crop=True, g_noise=True, rotate=True):
        self.flip     = flip
        self.r_crop   = r_crop
        self.g_noise  = g_noise
        self.rotate   = rotate
        print("holizontal flip : {}, random crop : {}, gaussian noise : {}, rotate : {}".format(
            self.flip, self.r_crop, self.g_noise, self.rotate,
        ))

    def __call__(self, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        already = False
        if self.flip and random.random() > 0.5:
            x = x.flip(-1)
            y = y.flip(-1)
            already = True

        if self.r_crop and random.random() > 0.5 and already:
            b, c, h, w = x.shape
            l, t = random.randint(0, h//2), random.randint(0,w//2)
            l1, t1 = random.randint(h//2, h), random.randint(w//2, w)
            x    = x[:,:,t:t+t1,l:l+l1]
            y    = y[:,:,t:t+t1,l:l+l1]

            x = TF.resize(x, size=[h, w], interpolation=T.InterpolationMode.NEAREST)
            y = TF.resize(y, size=[h, w], interpolation=T.InterpolationMode.NEAREST)
            already = True

        if self.g_noise and random.random() > 0.5 and already:
            n = torch.randn_like(x) * 0.15
            x = n + x
            already = True

        if self.rotate and random.random() > 0.5 and already:
            angle = random.randint(-30, 30)
            x = TF.rotate(x, angle)
            y = TF.rotate(y, angle)

        return x[0], y[0]
