import torch
import torch.nn as nn
import torch.optim as optim

class SelfAttentionSubNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionSubNet, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):

        x = x.unsqueeze(1)  # (batch_size, seq_len=1, input_dim)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output.squeeze(1)
        return self.hidden_layers(x)



class DynamicRouting(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(DynamicRouting, self).__init__()

        self.routing_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.routing_weights(x)


class DynamicLayeredOutput(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(DynamicLayeredOutput, self).__init__()

        self.outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])

    def forward(self, x, routing_weights):

        all_outputs = torch.stack([output(x) for output in self.outputs], dim=-1)  # (batch_size, output_dim=1, num_layers)

        final_output = torch.sum(all_outputs * routing_weights.unsqueeze(1), dim=-1)  # (batch_size, output_dim=1)
        return final_output.squeeze(-1)

class WeightedFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, *inputs):

        weighted_inputs = [w * x for w, x in zip(self.weights, inputs)]
        fused_output = torch.cat(weighted_inputs, dim=-1)
        return self.fc(fused_output)


class MultiModalNet(nn.Module):
    def __init__(self, feature_dims, hidden_dim, num_layers):
        super(MultiModalNet, self).__init__()

        self.subnets = nn.ModuleList([
            SelfAttentionSubNet(input_dim, hidden_dim) for input_dim in feature_dims
        ])

        self.fusion = WeightedFusion(hidden_dim * len(feature_dims), hidden_dim)
        self.routing_layer = DynamicRouting(hidden_dim, num_layers)
        self.layered_output = DynamicLayeredOutput(hidden_dim, num_layers)


    def forward(self, *inputs):

        sub_outputs = [subnet(x) for subnet, x in zip(self.subnets, inputs)]

        fused_output = abs(self.fusion(*sub_outputs))
        routing_weights = self.routing_layer(fused_output)
        final_output = abs(self.layered_output(fused_output, routing_weights))
        return final_output


# Define loss and optimizer
def create_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.002)

def create_criterion():
    return nn.MSELoss()

def create_model():
    model = MultiModalNet(feature_dims=[14, 14, 6], hidden_dim=16, num_layers=9)
    return model

