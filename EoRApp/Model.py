import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cpu')

# st.set_page_config(page_title='EoR App', page_icon=':smile:')

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, in_channels=1, emb_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (48 // patch_size) ** 3
        self.proj = nn.Linear(patch_size**3 * in_channels, emb_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

    def forward(self, x):
        # x shape: (n, 1, 48, 48, 48)
        n, c, d, h, w = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        x = x.contiguous().view(n, -1, c*self.patch_size**3)
        # x = x.permute(0, 2, 3, 1).contiguous().view(n, -1, self.patch_size**3 * c)
        x = self.proj(x)
        x = x + self.pos_embed
        return x
    
# output1 = PatchEmbedding()(input_cube)
# print(output1.shape)

class ParameterEmbedding(nn.Module):
    def __init__(self, param_dim=3, emb_dim=128):
        super(ParameterEmbedding, self).__init__()
        self.fc = nn.Linear(param_dim, emb_dim)

    def forward(self, params):
        # params shape: (n, 3)
        param_emb = self.fc(params)
        return param_emb.unsqueeze(1)  # shape: (n, 1, emb_dim)
    
# output2 = ParameterEmbedding()(input_params)
# print(output2.shape)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, ff_dim=256):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads)
        self.fc1 = nn.Linear(emb_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.fc2(F.relu(self.fc1(x)))
        x = x + ff_output
        x = self.norm2(x)
        return x

# output3 = TransformerEncoderLayer()(output1)
# print(output3.shape)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, param_dim=3, base_filters=32):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.enc2 = DoubleConv(base_filters, base_filters * 1)
        self.enc3 = DoubleConv(base_filters * 1, base_filters * 2)
        self.enc4 = DoubleConv(base_filters * 2, base_filters * 4)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck (where we will integrate parameters)
        self.bottleneck = DoubleConv(131, base_filters * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_filters * 10, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 5, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 3, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 1, base_filters)
        
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, params):
        
        # Encoder
        enc1 = self.enc1(x)   # shape: (n, 32, 48, 48, 48)
        enc2 = self.enc2(self.pool(enc1))  # shape: (n, 64, 24, 24, 24)
        enc3 = self.enc3(self.pool(enc2))  # shape: (n, 128, 12, 12, 12)
        enc4 = self.enc4(self.pool(enc3))  # shape: (n, 256, 6, 6, 6)

        # Integrating input parameters in bottleneck
        n, c, d, h, w = enc4.shape
        params = params.view(n, -1, 1, 1, 1).repeat(1, 1, d, h, w)
        bottleneck_input = torch.cat([enc4, params], dim=1)  # Concatenating params to feature maps
        bottleneck = self.bottleneck(bottleneck_input)  # shape: (n, 512, 6, 6, 6)
        
        # # Decoder
        dec4 = self.upconv4(bottleneck)  # shape: (n, 256, 12, 12, 12)
        dec4 = self.dec4(torch.cat([dec4, enc3], dim=1))
        
        dec3 = self.upconv3(dec4)  # shape: (n, 128, 24, 24, 24)
        dec3 = self.dec3(torch.cat([dec3, enc2], dim=1))
        
        dec2 = self.upconv2(dec3)  # shape: (n, 64, 48, 48, 48)
        dec2 = self.dec2(torch.cat([dec2, enc1], dim=1))
        
        dec1 = self.upconv1(dec2)  # shape: (n, 32, 48, 48, 48)
        dec1 = self.dec1(dec1)
        
        output = self.final_conv(dec1)  # shape: (n, 1, 48, 48, 48)
        return output


class CosmoUiT(nn.Module):
    def __init__(self, patch_size=6, emb_dim=128, num_heads=4, num_layers=4, param_dim=3):
        super(CosmoUiT, self).__init__()
        self.patch_embed = PatchEmbedding(patch_size, emb_dim=emb_dim)
        self.param_embed = ParameterEmbedding(param_dim, emb_dim)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(emb_dim, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(emb_dim, patch_size**3)
        self.unet = UNet3D()

    def forward(self, x, params):
        x = self.patch_embed(x)
        param_emb = self.param_embed(params)
        x = torch.cat((param_emb, x), dim=1)  # Concatenating parameter embedding
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.fc_out(x)  # shape: (n, num_patches, patch_size^3)
        x = self.reconstruct(x[:, 1:217, :])
        x = self.unet(x, params)
        return x
    
    def reconstruct(self, x):
        n, num_patches, patch_dim = x.shape
        patch_size = int(round(patch_dim ** (1/3)))
        d = h = w = int(round(num_patches ** (1/3)))
        x = x.view(n, d, h, w, patch_size, patch_size, patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(n, 1, d*patch_size, h*patch_size, w*patch_size)  # shape: (n, 1, 48, 48, 48)
        return x
    
# # Example usage
model = CosmoUiT(patch_size=8, emb_dim=128, num_heads=8, num_layers=4, param_dim=3).to(device)

model.load_state_dict(torch.load('CosmoUiTE1000.pth', map_location=torch.device('cpu'),weights_only=True))
model.eval() 

# dm = np.load('/media/disk1/prasad/codes/Data/ReducedDM.npy')
halo = np.load('ReducedHalo.npy')

# norm_dm = (dm - np.mean(dm))/np.std(dm)
norm_halo = (halo - np.mean(halo))/ np.std(halo)


# input_dm = torch.tensor(np.expand_dims(norm_dm, axis=[0,1]), dtype=torch.float32)
input_halo = torch.tensor(np.expand_dims(norm_halo, axis=[0,1]), dtype=torch.float32).to(device)


st.sidebar.header("EoR Parameters")
mh_min = st.sidebar.slider(
    "Minimum Halo Mass (Mh_min) [in units of M☉]",
    min_value=10.0,
    max_value=800.0,
    value=100.0,  # Default value
    step=1.0
)

nion = st.sidebar.slider(
    "Number of Ionizing Photons Produced per Baryon (Nion)",
    min_value=10.0,
    max_value=200.0,
    value=50.0,  # Default value
    step=1.0
)

rmfp = st.sidebar.slider(
    "Maximal Distance Travelled by Ionizing Photons (Rmfp) [in units of Mpc]",
    min_value=1.12,
    max_value=40.32,
    value=10.0,  # Default value
    step=0.1
)


parameters_tensor = torch.tensor([[mh_min, nion, rmfp]]).to(device)


with torch.no_grad():
    prediction = model(input_halo, parameters_tensor).detach().numpy()

# Let's take a slice of the cube for visualization (e.g., z = 24)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_halo.numpy()[0, 0, 24, :, :], cmap='viridis')
plt.title("Input Halo Mass - Slice (z = 24)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(prediction[0, 0, 24, :, :], cmap='viridis', vmin=0.0, vmax=1.0)
plt.title("Predicted Output - Slice (z = 24)")
plt.colorbar()

st.pyplot(plt)

