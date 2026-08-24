import torch
import torch.nn as nn
import torch.nn.functional as F


decom_num_layers = 3
kernel_size = 3
mid_dim = 5


class Decompose(nn.Module):
    def __init__(self, mid_dim):
        super().__init__()
        self.num_layers = decom_num_layers
        self.kernel_size = kernel_size
        self.mid_dim = mid_dim

        self.modulelist_u = nn.ModuleList()
        self.modulelist_d = nn.ModuleList()
        self.theta_list = nn.ParameterList()

        self.modulelist_u.append(nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(self.kernel_size // 2),
                nn.Conv2d(self.mid_dim, self.mid_dim, self.kernel_size, stride=1, bias=False),
            )
            for _ in range(4)
        ]))
        self.theta_list.append(nn.Parameter(torch.Tensor([0.0])))

        for _ in range(self.num_layers):
            self.modulelist_u.append(nn.ModuleList([
                nn.Sequential(
                    nn.ReflectionPad2d(self.kernel_size // 2),
                    nn.Conv2d(self.mid_dim, self.mid_dim, self.kernel_size, stride=1, bias=False),
                )
                for _ in range(4)
            ]))
            self.modulelist_d.append(nn.ModuleList([
                nn.Sequential(
                    nn.ReflectionPad2d(self.kernel_size // 2),
                    nn.Conv2d(self.mid_dim, self.mid_dim, self.kernel_size, stride=1, bias=False),
                )
                for _ in range(4)
            ]))
            self.theta_list.append(nn.Parameter(torch.Tensor([0.0])))

    def _dim_down(self, i, w):
        x = w[:, 0:self.mid_dim]
        y = w[:, self.mid_dim:2 * self.mid_dim]
        c = w[:, 2 * self.mid_dim:3 * self.mid_dim]
        p0 = self.modulelist_d[i][0](x) + self.modulelist_d[i][2](c)
        p1 = self.modulelist_d[i][1](y) + self.modulelist_d[i][3](c)
        return torch.cat((p0, p1), dim=1)

    def _dim_up(self, i, z):
        x = z[:, 0:self.mid_dim]
        y = z[:, self.mid_dim:2 * self.mid_dim]
        p0 = self.modulelist_u[i][0](x)
        p1 = self.modulelist_u[i][1](y)
        p2 = self.modulelist_u[i][2](x) + self.modulelist_u[i][3](y)
        return torch.cat((p0, p1, p2), dim=1)

    def forward(self, x):
        p0 = self._dim_up(0, x)
        w = torch.sign(p0) * F.relu(torch.abs(p0) - self.theta_list[0])
        for i in range(self.num_layers):
            p1 = self._dim_down(i, w)
            p2 = p1 - x
            p3 = self._dim_up(i + 1, p2)
            p4 = w - p3
            w = torch.sign(p4) * F.relu(torch.abs(p4) - self.theta_list[i + 1])
        return w


class Fuse(nn.Module):
    def __init__(self, mid_dim):
        super().__init__()
        self.mid_dim = mid_dim
        self.conv1 = nn.Conv2d(3 * self.mid_dim, 2 * self.mid_dim, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(2 * self.mid_dim, self.mid_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, w, z):
        p0 = self.conv1(w) + z
        return self.conv2(p0)


class CDNet(nn.Module):
    """Two-input CDNet fusion model.

    Inputs use BHWC format, one channel, float range [0, 1].
    Output uses BHWC format, one channel, float range [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.mid_dim = mid_dim
        self.head_transform = nn.Conv2d(2, 2 * self.mid_dim, kernel_size=1, stride=1, bias=False)
        self.dec = Decompose(self.mid_dim)
        self.fuse = Fuse(self.mid_dim)
        self.tail_transform = nn.Conv2d(self.mid_dim, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x, y):
        x = x.permute(0, 3, 1, 2).float()
        y = y.permute(0, 3, 1, 2).float()
        z = torch.cat((x, y), dim=1)
        z = self.head_transform(z)
        w = self.dec(z)
        fused = self.fuse(w, z)
        fused = self.tail_transform(fused)
        return fused.permute(0, 2, 3, 1).clamp(0, 1)
