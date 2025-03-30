import torch


def Conv2dSame(in_channels, out_channels, kernel_size, use_bias=True, padding_layer=torch.nn.ReflectionPad2d):
    ka = kernel_size // 2
    kb = ka - 1 if kernel_size % 2 == 0 else ka
    return [
        padding_layer((ka, kb, ka, kb)),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=use_bias)
    ]


def conv2d_bn(in_channels, filters, kernel_size, padding='same', activation='relu'):
    assert padding == 'same'
    affine = False if activation == 'relu' or activation == 'sigmoid' else True
    sequence = []
    sequence += Conv2dSame(in_channels, filters, kernel_size, use_bias=False)
    sequence += [torch.nn.BatchNorm2d(filters, affine=affine)]
    if activation == "relu":
        sequence += [torch.nn.ReLU()]
    elif activation == "sigmoid":
        sequence += [torch.nn.Sigmoid()]
    elif activation == 'tanh':
        sequence += [torch.nn.Tanh()]
    return torch.nn.Sequential(*sequence)


class MultiResBlock(torch.nn.Module):
    def __init__(self, in_channels, u, alpha=1.67, use_dropout=False):
        super().__init__()
        w = alpha * u
        self.out_channel = int(w * 0.167) + int(w * 0.333) + int(w * 0.5)
        self.conv2d_bn = conv2d_bn(in_channels, self.out_channel, 1, activation=None)
        self.conv3x3 = conv2d_bn(in_channels, int(w * 0.167), 3, activation='relu')
        self.conv5x5 = conv2d_bn(int(w * 0.167), int(w * 0.333), 3, activation='relu')
        self.conv7x7 = conv2d_bn(int(w * 0.333), int(w * 0.5), 3, activation='relu')
        self.bn_1 = torch.nn.BatchNorm2d(self.out_channel)
        self.relu = torch.nn.ReLU()
        self.bn_2 = torch.nn.BatchNorm2d(self.out_channel)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = torch.nn.Dropout(0.5)

    def forward(self, inp):
        if self.use_dropout:
            x = self.dropout(inp)
        else:
            x = inp
        shortcut = self.conv2d_bn(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = self.bn_1(out)
        out = torch.add(shortcut, out)
        out = self.relu(out)
        out = self.bn_2(out)
        return out


class ResPathBlock(torch.nn.Module):
    def __init__(self, in_channels, filters):
        super(ResPathBlock, self).__init__()
        self.conv2d_bn1 = conv2d_bn(in_channels, filters, 1, activation=None)
        self.conv2d_bn2 = conv2d_bn(in_channels, filters, 3, activation='relu')
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(filters)

    def forward(self, inp):
        shortcut = self.conv2d_bn1(inp)
        out = self.conv2d_bn2(inp)
        out = torch.add(shortcut, out)
        out = self.relu(out)
        out = self.bn(out)
        return out


class ResPath(torch.nn.Module):
    def __init__(self, in_channels, filters, length):
        super(ResPath, self).__init__()
        self.first_block = ResPathBlock(in_channels, filters)
        self.blocks = torch.nn.Sequential(*[ResPathBlock(filters, filters) for i in range(length - 1)])

    def forward(self, inp):
        out = self.first_block(inp)
        out = self.blocks(out)
        return out


class MultiResUnet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, nf=32, use_dropout=False):
        super(MultiResUnet, self).__init__()
        self.mres_block1 = MultiResBlock(in_channels, u=nf)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.res_path1 = ResPath(self.mres_block1.out_channel, nf, 4)

        self.mres_block2 = MultiResBlock(self.mres_block1.out_channel, u=nf * 2)
        # self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.res_path2 = ResPath(self.mres_block2.out_channel, nf * 2, 3)

        self.mres_block3 = MultiResBlock(self.mres_block2.out_channel, u=nf * 4)
        # self.pool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.res_path3 = ResPath(self.mres_block3.out_channel, nf * 4, 2)

        self.mres_block4 = MultiResBlock(self.mres_block3.out_channel, u=nf * 8)
        # self.pool4 = torch.nn.MaxPool2d(kernel_size=2)
        self.res_path4 = ResPath(self.mres_block4.out_channel, nf * 8, 1)

        self.mres_block5 = MultiResBlock(self.mres_block4.out_channel, u=nf * 16)

        self.deconv1 = torch.nn.ConvTranspose2d(self.mres_block5.out_channel, nf * 8, (2, 2), (2, 2))
        self.mres_block6 = MultiResBlock(nf * 8 + nf * 8, u=nf * 8, use_dropout=use_dropout)
        # MultiResBlock(nf * 8 + self.mres_block4.out_channel, u=nf * 8)

        self.deconv2 = torch.nn.ConvTranspose2d(self.mres_block6.out_channel, nf * 4, (2, 2), (2, 2))
        self.mres_block7 = MultiResBlock(nf * 4 + nf * 4, u=nf * 4, use_dropout=use_dropout)
        # MultiResBlock(nf * 4 + self.mres_block3.out_channel, u=nf * 4)

        self.deconv3 = torch.nn.ConvTranspose2d(self.mres_block7.out_channel, nf * 2, (2, 2), (2, 2))
        self.mres_block8 = MultiResBlock(nf * 2 + nf * 2, u=nf * 2, use_dropout=use_dropout)
        # MultiResBlock(nf * 2 + self.mres_block2.out_channel, u=nf * 2)

        self.deconv4 = torch.nn.ConvTranspose2d(self.mres_block8.out_channel, nf, (2, 2), (2, 2))
        self.mres_block9 = MultiResBlock(nf + nf, u=nf)
        # MultiResBlock(nf + self.mres_block1.out_channel, u=nf)

        self.conv10 = conv2d_bn(self.mres_block9.out_channel, out_channels, 1, padding='same', activation='tanh')

    def forward(self, inp):
        mresblock1 = self.mres_block1(inp)
        pool = self.pool(mresblock1)
        mresblock1 = self.res_path1(mresblock1)

        mresblock2 = self.mres_block2(pool)
        pool = self.pool(mresblock2)
        mresblock2 = self.res_path2(mresblock2)

        mresblock3 = self.mres_block3(pool)
        pool = self.pool(mresblock3)
        mresblock3 = self.res_path3(mresblock3)

        mresblock4 = self.mres_block4(pool)
        pool = self.pool(mresblock4)
        mresblock4 = self.res_path4(mresblock4)

        mresblock = self.mres_block5(pool)

        up = torch.cat([self.deconv1(mresblock), mresblock4], dim=1)
        mresblock = self.mres_block6(up)

        up = torch.cat([self.deconv2(mresblock), mresblock3], dim=1)
        mresblock = self.mres_block7(up)

        up = torch.cat([self.deconv3(mresblock), mresblock2], dim=1)
        mresblock = self.mres_block8(up)

        up = torch.cat([self.deconv4(mresblock), mresblock1], dim=1)
        mresblock = self.mres_block9(up)

        conv10 = self.conv10(mresblock)
        return conv10


class MultiResUnetGenerator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, gpu_ids=[]):
        super(MultiResUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.model = MultiResUnet(input_nc, output_nc, nf=ngf, use_dropout=use_dropout)

    def forward(self, inp):
        if self.gpu_ids and isinstance(inp.data, torch.cuda.FloatTensor):
            return torch.nn.parallel.data_parallel(self.model, inp, self.gpu_ids)
        else:
            return self.model(inp)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname == 'Conv2d':
        pass
    print(classname)


# a = ResPath(10, 100,3)
# a.apply(weights_init_uniform_rule)
#a = MultiResUnet(512, 512, 3)
#x = torch.randn(2, 3, 512, 512)
#print(a(x).shape)
