import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

from .gat_conv import GATConv


class STAGATE(torch.nn.Module):
    def __init__(self, in_channels):
        super(STAGATE, self).__init__()

        self.conv1 = GATConv(
            in_channels,
            512,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv2 = GATConv(
            512, 30, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False
        )
        self.conv3 = GATConv(
            30, 512, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False
        )
        self.conv4 = GATConv(
            512,
            in_channels,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x1 = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x2 = F.elu(self.conv2(x1, edge_index))
        x2 = self.conv2(x1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        x3 = F.elu(
            self.conv3(
                x2, edge_index, attention=True, tied_attention=self.conv1.attentions
            )
        )
        x4 = self.conv4(x3, edge_index, attention=False)

        return x2, x4  # F.log_softmax(x, dim=-1)
