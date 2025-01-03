import torch
import torch.nn as nn
import torch.nn.functional as F
import math

gks = 5
pad = [i for i in range(gks*gks)]
shift = torch.zeros(gks*gks, 4)
for i in range(gks):
    for j in range(gks):
        top = i
        bottom = gks-1-i
        left = j
        right = gks-1-j
        pad[i*gks + j] = torch.nn.ZeroPad2d((left, right, top, bottom))
        #shift[i*gks + j, :] = torch.tensor([left, right, top, bottom])
mid_pad = torch.nn.ZeroPad2d(((gks-1)/2, (gks-1)/2, (gks-1)/2, (gks-1)/2))
zero_pad = pad[0]

gks2 = 3     #guide kernel size
pad2 = [i for i in range(gks2*gks2)]
shift = torch.zeros(gks2*gks2, 4)
for i in range(gks2):
    for j in range(gks2):
        top = i
        bottom = gks2-1-i
        left = j
        right = gks2-1-j
        pad2[i*gks2 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

gks3 = 7     #guide kernel size
pad3 = [i for i in range(gks3*gks3)]
shift = torch.zeros(gks3*gks3, 4)
for i in range(gks3):
    for j in range(gks3):
        top = i
        bottom = gks3-1-i
        left = j
        right = gks3-1-j
        pad3[i*gks3 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)


    
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            #norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)

class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result

class CSPNGenerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)

        #padding
        weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size*self.kernel_size):
            zero_pad = 0
            if(self.kernel_size==3):
                zero_pad = pad2[t]
            elif(self.kernel_size==5):
                zero_pad = pad[t]
            elif(self.kernel_size==7):
                zero_pad = pad3[t]
            if(t < int((self.kernel_size*self.kernel_size-1)/2)):
                weight_pad[t] = zero_pad(guide[:, t:t+1, :, :])
            elif(t > int((self.kernel_size*self.kernel_size-1)/2)):
                weight_pad[t] = zero_pad(guide[:, t-1:t, :, :])
            else:
                weight_pad[t] = zero_pad(guide_mid)

        guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size*self.kernel_size)], dim=1)
        return guide_weight

class CSPN(nn.Module):
  def __init__(self, kernel_size):
      super(CSPN, self).__init__()
      self.kernel_size = kernel_size

  def forward(self, guide_weight, hn, h0):

        #CSPN
        half = int(0.5 * (self.kernel_size * self.kernel_size - 1))
        result_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size*self.kernel_size):
            zero_pad = 0
            if(self.kernel_size==3):
                zero_pad = pad2[t]
            elif(self.kernel_size==5):
                zero_pad = pad[t]
            elif(self.kernel_size==7):
                zero_pad = pad3[t]
            if(t == half):
                result_pad[t] = zero_pad(h0)
            else:
                result_pad[t] = zero_pad(hn)
        guide_result = torch.cat([result_pad[t] for t in range(self.kernel_size*self.kernel_size)], dim=1)
        #guide_result = torch.cat([result0_pad, result1_pad, result2_pad, result3_pad,result4_pad, result5_pad, result6_pad, result7_pad, result8_pad], 1)

        guide_result = torch.sum((guide_weight.mul(guide_result)), dim=1)
        guide_result = guide_result[:, int((self.kernel_size-1)/2):-int((self.kernel_size-1)/2), int((self.kernel_size-1)/2):-int((self.kernel_size-1)/2)]

        return guide_result.unsqueeze(dim=1)

class CSPNGenerateAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization in standard CSPN
        #'''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        #'''
        #weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output =  torch.cat((half1, guide_mid, half2), dim=1)
        return output

def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size-1)/2))
    return kernel

class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0): #with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+1, :] = input0

        #print(input_im2col.size(), kernel.size())
        output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel))
        return output.view(bs, 1, h, w)

class GeometryFeature(nn.Module):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z*(0.5*h*(vnorm+1)-ch)/fh
        y = z*(0.5*w*(unorm+1)-cw)/fw
        return torch.cat((x, y, z),1)

class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            #norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# class MRConv2d(nn.Module):
#     """
#     Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
#         super(MRConv2d, self).__init__()
#         self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

#     def forward(self, x, edge_index, y=None):
#         x_i = batched_index_select(x, edge_index[1])
#         if y is not None:
#             x_j = batched_index_select(y, edge_index[0])
#         else:
#             x_j = batched_index_select(x, edge_index[0])
#         x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
#         b, c, n, _ = x.shape
#         x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
#         return self.nn(x)


# class EdgeConv2d(nn.Module):
#     """
#     Edge convolution layer (with activation, batch normalization) for dense data type
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
#         super(EdgeConv2d, self).__init__()
#         self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

#     def forward(self, x, edge_index, y=None):
#         x_i = batched_index_select(x, edge_index[1])
#         if y is not None:
#             x_j = batched_index_select(y, edge_index[0])
#         else:
#             x_j = batched_index_select(x, edge_index[0])
#         max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
#         return max_value


# class GraphSAGE(nn.Module):
#     """
#     GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
#         super(GraphSAGE, self).__init__()
#         self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
#         self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

#     def forward(self, x, edge_index, y=None):
#         if y is not None:
#             x_j = batched_index_select(y, edge_index[0])
#         else:
#             x_j = batched_index_select(x, edge_index[0])
#         x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
#         return self.nn2(torch.cat([x, x_j], dim=1))


# class GINConv2d(nn.Module):
#     """
#     GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
#     """
#     def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
#         super(GINConv2d, self).__init__()
#         self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
#         eps_init = 0.0
#         self.eps = nn.Parameter(torch.Tensor([eps_init]))

#     def forward(self, x, edge_index, y=None):
#         if y is not None:
#             x_j = batched_index_select(y, edge_index[0])
#         else:
#             x_j = batched_index_select(x, edge_index[0])
#         x_j = torch.sum(x_j, -1, keepdim=True)
#         return self.nn((1 + self.eps) * x + x_j)


# class GraphConv2d(nn.Module):
#     """
#     Static graph convolution layer
#     """
#     def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
#         super(GraphConv2d, self).__init__()
#         if conv == 'edge':
#             self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
#         elif conv == 'mr':
#             self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
#         elif conv == 'sage':
#             self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
#         elif conv == 'gin':
#             self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
#         else:
#             raise NotImplementedError('conv:{} is not supported'.format(conv))

#     def forward(self, x, edge_index, y=None):
#         return self.gconv(x, edge_index, y)


# class GraphConv2d(nn.Module):
#     """
#     Static graph convolution layer
#     """
#     def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
#         super(GraphConv2d, self).__init__()
#         if conv == 'edge':
#             self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
#         elif conv == 'mr':
#             self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
#         elif conv == 'sage':
#             self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
#         elif conv == 'gin':
#             self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
#         else:
#             raise NotImplementedError('conv:{} is not supported'.format(conv))

#     def forward(self, x, edge_index, y=None):
#         return self.gconv(x, edge_index, y)
    
# class DyGraphConv2d(GraphConv2d):
#     """
#     Dynamic graph convolution layer
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
#                  norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
#         super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
#         self.k = kernel_size
#         self.d = dilation
#         self.r = r
#         self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

#     def forward(self, x, relative_pos=None):
#         B, C, H, W = x.shape
#         y = None
#         if self.r > 1:
#             y = F.avg_pool2d(x, self.r, self.r)
#             y = y.reshape(B, C, -1, 1).contiguous()            
#         x = x.reshape(B, C, -1, 1).contiguous()
#         edge_index = self.dilated_knn_graph(x, y, relative_pos)
#         x = super(DyGraphConv2d, self).forward(x, edge_index, y)
#         return x.reshape(B, -1, H, W).contiguous()
    
# class Grapher(nn.Module):
#     """
#     Grapher module with graph convolution and fc layers
#     """
#     def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
#                  bias=True,  stochastic=False, epsilon=0.2, r=1, n=196, drop_path=0.0, relative_pos=False):
#         super(Grapher, self).__init__()
#         self.channels = in_channels
#         self.n = n
#         self.r = r
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#         )
#         self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
#                               act, norm, bias, stochastic, epsilon, r)
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.relative_pos = None
#         if relative_pos:
#             print('using relative_pos')
#             relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
#                 int(n**0.5)))).unsqueeze(0).unsqueeze(1)
#             relative_pos_tensor = F.interpolate(
#                     relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
#             self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

#     def _get_relative_pos(self, relative_pos, H, W):
#         if relative_pos is None or H * W == self.n:
#             return relative_pos
#         else:
#             N = H * W
#             N_reduced = N // (self.r * self.r)
#             return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

#     def forward(self, x):
#         _tmp = x
#         x = self.fc1(x)
#         B, C, H, W = x.shape
#         relative_pos = self._get_relative_pos(self.relative_pos, H, W)
#         x = self.graph_conv(x, relative_pos)
#         x = self.fc2(x)
#         x = self.drop_path(x) + _tmp
#         return x

# class FFN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
#             nn.BatchNorm2d(hidden_features),
#         )
#         self.act = act_layer(act)
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
#             nn.BatchNorm2d(out_features),
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         shortcut = x
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         x = self.drop_path(x) + shortcut
#         return x

