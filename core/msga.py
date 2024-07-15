import torch
import torch.nn as nn
from loss import batch_episym
from einops import rearrange


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]
    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    x_shp = x_in.shape
    x_in = x_in.squeeze(1)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class AnnularConv(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(AnnularConv, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        out = self.conv(features)
        return out


class Cross_transformer(nn.Module):
    def __init__(self, channel):
        super(Cross_transformer, self).__init__()
        self.attq = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.attk = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.mlp = nn.Conv2d(channel, channel, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, n1, n2, q_1):
        q = self.attq(n1)
        k = self.attk(n2)
        v = self.attv(n2)
        att = torch.mul(q, k)
        score = torch.softmax(att, dim=3)
        add_value = torch.mul(score, v)
        add_value = self.mlp(add_value)
        out = add_value.max(dim=-1, keepdim=False)[0]
        out = self.relu(out)
        out = out.unsqueeze(-1) + q_1
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(0.)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))
        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class SE_Block(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl * xg
        wei = self.sigmoid(xlg)
        out = wei * x
        return out


class Topk_transformer(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.attn = Attention(channels, num_heads=4)
        self.se_block = SE_Block(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm)
        x_se = self.se_block(x_norm)
        x = x + x_se + x_attn
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2))
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down, S).unsqueeze(3)
        return out


class LayerBlock(nn.Module):
    def __init__(self, channels, grid=16):
        nn.Module.__init__(self)
        self.output = grid * grid
        self.upsample = diff_pool(channels, self.output)
        self.grid_num = grid
        self.glo_exchange1 = Topk_transformer(channels)
        self.glo_exchange2 = Topk_transformer(channels)
        self.downsample = diff_unpool(channels, self.output)

    def forward(self, x):
        x_up = self.upsample(x)
        x_up = x_up.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        x_glo = self.glo_exchange1(x_up)
        x_glo = self.glo_exchange2(x_glo)
        x_glo = x_glo.reshape(x.shape[0], x.shape[1], self.grid_num * self.grid_num)
        x = self.downsample(x, x_glo)
        return x


class GloBlock(nn.Module):
    def __init__(self, channels, k_num, grid=16):
        nn.Module.__init__(self)
        self.transformer1 = LayerBlock(channels, grid)

    def forward(self, x):
        out = self.transformer1(x)
        return out


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.resnet1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.aggregation1 = AnnularConv(self.k_num, self.out_channel)
        self.resnet2 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.aggregation2 = AnnularConv(self.k_num, self.out_channel)
        self.fusion = Cross_transformer(self.out_channel)
        self.glo_block = GloBlock(self.out_channel, k_num=self.k_num)
        self.fusion1 = Cross_transformer(self.out_channel)
        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2,
                                       index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out)
        out = self.resnet1(out)
        graph1 = get_graph_feature(out, k=self.k_num)
        out1 = self.aggregation1(graph1)
        out2 = self.resnet2(out1)
        graph2 = get_graph_feature(out2, k=self.k_num)
        out = self.fusion(graph1, graph2, out)
        w0 = self.linear_0(out).view(B, -1)
        out_g = self.glo_block(out)
        out = out_g + out
        graph3 = get_graph_feature(out, k=self.k_num)
        out = self.fusion1(graph2, graph3, out2)
        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class MSGANet(nn.Module):
    def __init__(self, config):
        super(MSGANet, self).__init__()
        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9,
                             sampling_rate=config.sr)
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        B, _, N, _ = x.shape
        x1, y1, ws0, w_ds0 = self.ds_0(x, y)
        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)
        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)
        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

