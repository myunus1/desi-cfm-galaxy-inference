import torch
import torch.nn as nn
from torch.nn import functional as F

# class ImageEncoder(nn.Module):
#     """
#     A simple CNN-based image encoder that extracts feature vectors from galaxy images.
#     Input: x of shape (batch_size, 3, H, W)
#     Output: features of shape (batch_size, feature_dim)
#     """
#     def __init__(self, feature_dim=128):
#         super(ImageEncoder, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, H/2, W/2)
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, H/4, W/4)
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> (128, H/8, W/8)
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (256, H/16, W/16)
#             nn.ReLU(),
#         )
#         # flatten and project to feature_dim
#         self.fc = nn.Linear(256 * 8 * 8, feature_dim)

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         features = self.fc(x)
#         return features

# class ImageEncoder(nn.Module):
#     """
#     CNN-based image encoder with less aggressive downsampling.
#     Input:  x of shape (batch_size, 3, 128, 128)
#     Output: features of shape (batch_size, feature_dim)
#     """
#     def __init__(self, feature_dim=128):
#         super(ImageEncoder, self).__init__()
#         self.conv_layers = nn.Sequential(
#             # 128×128 → 64×64
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             # nn.AvgPool2d(2, 2), 

#             # 64×64 → 32×32
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             # nn.AvgPool2d(2, 2), 

#             # 32×32 → 16×16
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             # nn.AvgPool2d(2, 2), 

#             # keep 16×16 spatial resolution
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         )
#         # flatten (256 × 16 × 16) → feature_dim
#         self.fc = nn.Linear(256 * 16 * 16, feature_dim)

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)    # (batch, 256*16*16)
#         features = self.fc(x)
#         return features

# class ImageEncoder(nn.Module):
#     def __init__(self, feature_dim=128):
#         super().__init__()
#         self.enc = nn.Sequential(
#             # — No information lost in conv itself —
#             nn.Conv2d(3, 32, 3, stride=1, padding=1),  # -> 128×128
#             nn.ReLU(),
#             # nn.MaxPool2d(2),                           # -> 64×64
#             nn.AvgPool2d(2),

#             nn.Conv2d(32, 64, 3, stride=1, padding=1), # -> 64×64
#             nn.ReLU(),
#             # nn.MaxPool2d(2),                           # -> 32×32
#             nn.AvgPool2d(2),

#             # keep stride=1 here to preserve detail at 32×32
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),# -> 32×32
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 3, stride=1, padding=1),# -> 32×32
#             nn.ReLU(),
#         )
#         # globalize features without flattening 32×32 into huge vector
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))       # -> 1×1
#         self.fc   = nn.Linear(256, feature_dim)

#     def forward(self, x):
#         x = self.enc(x)
#         x = self.pool(x).view(x.size(0), -1)  # [B, 256]
#         return self.fc(x)                     # [B, feature_dim]

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            # Conv -> ReLU -> Pool pattern throughout
            nn.Conv2d(3, 32, 3, stride=1, padding=1),   # -> 128×128
            nn.ReLU(),
            nn.AvgPool2d(2),                            # -> 64×64

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # -> 64×64
            nn.ReLU(),
            nn.AvgPool2d(2),                            # -> 32×32

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # -> 32×32
            nn.ReLU(),
            nn.AvgPool2d(2),                            # -> 16×16

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # -> 16×16
            nn.ReLU(),
            nn.AvgPool2d(2),                            # -> 8×8
        )
        # Global average pooling to get 256-dimensional features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))        # -> 1×1
        # No final projection layer - use the 256D features directly

    def forward(self, x):
        x = self.enc(x)                                 # [B, 256, 8, 8]
        x = self.pool(x).view(x.size(0), -1)          # [B, 256]
        return x                                        # [B, 256]
        
# class VelocityNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim=256, output_dim=None):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(self, t, x, img_feat):
#         # t: (batch,1), x:(batch,property_dim), img_feat:(batch,feature_dim)
#         inp = torch.cat([t, x, img_feat], dim=1)
#         return self.net(inp)


# class ConditionalFlowModel(nn.Module):
#     def __init__(self, property_dim, feature_dim=128, hidden_dim=256):
#         """
#         property_dim: number of tabular variables
#         feature_dim: size of the image‐encoder output
#         """
#         super().__init__()
#         self.encoder = ImageEncoder(feature_dim=feature_dim)
#         self.velocity_net = VelocityNet(
#             input_dim=1 + property_dim + feature_dim,
#             hidden_dim=hidden_dim,
#             output_dim=property_dim
#         )

#     def forward(self, t, x, img):
#         img_feat = self.encoder(img)
#         return self.velocity_net(t, x, img_feat)

class VelocityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x, cond_feat):
        # cond_feat is now [img_feat ; phot] of size feature_dim+phot_dim
        inp = torch.cat([t, x, cond_feat], dim=1)
        return self.net(inp)

class ConditionalFlowModel(nn.Module):
    def __init__(self,
                 property_dim,       # number of target vars, e.g. 5
                 feature_dim=128,    # image‐encoder output
                 phot_dim=5,         # photometry dim (u,g,r,i,z)
                 hidden_dim=256
                 ):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim=feature_dim)
        # new input_dim: 1 + property_dim + feature_dim + phot_dim
        self.velocity_net = VelocityNet(
            input_dim=1 + property_dim + feature_dim + phot_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim
        )

    def forward(self, t, x, img, phot):
        """
        t:    (batch,1)
        x:    (batch,property_dim)
        img:  (batch,3,H,W)
        phot: (batch,phot_dim)
        """
        img_feat  = self.encoder(img)                  # (batch,feature_dim)
        cond_feat = torch.cat([img_feat, phot], dim=1) # (batch,feature_dim+phot_dim)
        return self.velocity_net(t, x, cond_feat)

class ImprovedVelocityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x, cond_feat):
        # cond_feat is now [img_feat ; phot] of size feature_dim+phot_dim
        inp = torch.cat([t, x, cond_feat], dim=1)
        return self.net(inp)

class ImprovedConditionalFlowModel(nn.Module):
    """
    Image + Photometry CFM using the same ImageEncoder but with ImprovedVelocityNet
    (5-layer 512-wide MLP).  Drop-in replacement for ConditionalFlowModel with a
    deeper velocity network.
    """
    def __init__(self,
                 property_dim,
                 feature_dim=256,
                 phot_dim=5,
                 hidden_dim=512):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim=feature_dim)
        self.velocity_net = ImprovedVelocityNet(
            input_dim=1 + property_dim + feature_dim + phot_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim
        )

    def forward(self, t, x, img, phot):
        """
        t:    (batch,1)
        x:    (batch,property_dim)
        img:  (batch,3,H,W)
        phot: (batch,phot_dim)
        """
        img_feat  = self.encoder(img)
        cond_feat = torch.cat([img_feat, phot], dim=1)
        return self.velocity_net(t, x, cond_feat)

class PhotometryEncoder(nn.Module):
    """
    MLP to embed photometry into a feature vector of size `feature_dim`.
    """
    def __init__(self, phot_dim, feature_dim=128, hidden_dim=128):
        """
        phot_dim    : number of photometric channels (e.g. 5 for ugriz)
        hidden_dim  : size of the MLP hidden layer
        feature_dim : size of the output embedding
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, phot):
        # phot: (batch_size, phot_dim)
        return self.net(phot)


class PhotometryConditionalFlowModel(nn.Module):
    """
    Conditional Flow Matching model that conditions on photometry instead of an image.
    """
    def __init__(self, phot_dim, property_dim, feature_dim=128, hidden_dim=256):
        """
        phot_dim     : number of photometric channels (e.g. 5)
        property_dim : number of target variables
        feature_dim  : size of the photometry embedding
        hidden_dim   : hidden size of the velocity net
        """
        super().__init__()
        # 1) embed photometry into feature_dim vector
        self.encoder = PhotometryEncoder(
            phot_dim=phot_dim,
            feature_dim=feature_dim,
            hidden_dim=feature_dim  # you can adjust this if you like
        )
        # 2) velocity network exactly as in ConditionalFlowModel, but dynamic dims:
        self.velocity_net = VelocityNet(
            input_dim=1 + property_dim + feature_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim
        )

    def forward(self, t, x, phot):
        """
        t:    (batch, 1)           time
        x:    (batch, property_dim) current latent state
        phot: (batch, phot_dim)    normalized photometry
        """
        feat = self.encoder(phot)
        return self.velocity_net(t, x, feat)
    
class PhotometryConditionalFlowModelSimple(nn.Module):
    """
    Conditional Flow Matching model that conditions directly
    on the N-band photometry vector (no MLP encoder).
    """
    def __init__(self, phot_dim, property_dim, hidden_dim=256):
        """
        phot_dim     : number of photometric channels (e.g. 5 for ugriz)
        property_dim : number of target variables (e.g. 6)
        hidden_dim   : hidden size of the velocity net
        """
        super().__init__()
        # VelocityNet from your code; it concatenates (t, x, cond_feat)
        self.velocity_net = VelocityNet(
            input_dim=1 + property_dim + phot_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim
        )

    def forward(self, t: torch.Tensor,
                      x: torch.Tensor,
                      phot: torch.Tensor) -> torch.Tensor:
        """
        t   : (batch, 1)           time embedding
        x   : (batch, property_dim) current state
        phot: (batch, phot_dim)    normalized photometry
        """
        # Directly pass the 5-D photometry as the conditioning vector
        return self.velocity_net(t, x, phot)

class ImprovedPhotometryConditionalFlowModelSimple(nn.Module):
    """
    Conditional Flow Matching model that conditions directly
    on the N-band photometry vector (no MLP encoder).
    """
    def __init__(self, phot_dim, property_dim, hidden_dim=512):
        """
        phot_dim     : number of photometric channels (e.g. 5 for ugriz)
        property_dim : number of target variables (e.g. 6)
        hidden_dim   : hidden size of the velocity net
        """
        super().__init__()
        # VelocityNet from your code; it concatenates (t, x, cond_feat)
        self.velocity_net = ImprovedVelocityNet(
            input_dim=1 + property_dim + phot_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim
        )

    def forward(self, t: torch.Tensor,
                      x: torch.Tensor,
                      phot: torch.Tensor) -> torch.Tensor:
        """
        t   : (batch, 1)           time embedding
        x   : (batch, property_dim) current state
        phot: (batch, phot_dim)    normalized photometry
        """
        # Directly pass the 5-D photometry as the conditioning vector
        return self.velocity_net(t, x, phot)


class ResidualVelocityNet(nn.Module):
    """
    Velocity network that computes a base velocity from a frozen photometry model
    and adds a learned residual correction conditioned on image features.

    v_total(t, x, [img_feat; phot]) = v_base(t, x, phot) + v_residual(t, x, [img_feat; phot])

    This forces the residual network to learn only what photometry cannot capture
    (e.g., morphological information relevant to DN4000, spatially-resolved features).
    """
    def __init__(self, phot_velocity_net, input_dim, hidden_dim=256,
                 output_dim=None, phot_dim=5, feature_dim=256):
        """
        phot_velocity_net : nn.Module
            The velocity_net from a pretrained photometry-only model (will be frozen).
        input_dim         : int
            Input dimension for the residual net: 1 + property_dim + feature_dim + phot_dim.
        hidden_dim        : int
            Hidden layer size for the residual net.
        output_dim        : int
            Output dimension (= property_dim).
        phot_dim          : int
            Number of photometry channels, used to split cond_feat.
        feature_dim       : int
            Image encoder output dimension, used to split cond_feat.
        """
        super().__init__()
        # Frozen base photometry velocity net
        self.phot_velocity_net = phot_velocity_net
        for p in self.phot_velocity_net.parameters():
            p.requires_grad = False

        self.phot_dim = phot_dim
        self.feature_dim = feature_dim

        # Trainable residual correction network
        self.residual_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x, cond_feat):
        """
        t         : (batch, 1)
        x         : (batch, property_dim)
        cond_feat : (batch, feature_dim + phot_dim)  — [img_feat ; phot]
        """
        # Extract photometry from the tail of the conditioning vector
        phot = cond_feat[:, self.feature_dim:]

        # Base velocity from frozen photometry model (no grad)
        with torch.no_grad():
            v_base = self.phot_velocity_net(t, x, phot)

        # Residual velocity from image + photometry features
        inp = torch.cat([t, x, cond_feat], dim=1)
        v_residual = self.residual_net(inp)

        return v_base + v_residual


class ResidualConditionalFlowModel(nn.Module):
    """
    Residual Conditional Flow Matching model.

    Architecture:
        1. A frozen pretrained photometry model provides base velocity predictions.
        2. A trainable image encoder extracts morphological features.
        3. A trainable residual velocity net predicts corrections conditioned on
           both image features and photometry.
        4. Total velocity = base + residual.

    This design forces the image branch to learn complementary information that
    photometry alone cannot capture, making the improvement from images explicit
    and measurable (the residual term magnitude).

    Compatible with the existing sample_properties_distribution_rk4() sampler
    via .encoder and .velocity_net attributes.
    """
    def __init__(self, phot_model, property_dim, feature_dim=256,
                 phot_dim=5, hidden_dim=256):
        """
        phot_model   : nn.Module
            A pretrained PhotometryConditionalFlowModelSimple (will be frozen).
        property_dim : int
            Number of target variables (e.g. 5).
        feature_dim  : int
            Output dimension of the image encoder.
        phot_dim     : int
            Number of photometry channels.
        hidden_dim   : int
            Hidden layer size for the residual velocity net.
        """
        super().__init__()
        # Trainable image encoder
        self.encoder = ImageEncoder(feature_dim=feature_dim)

        # Residual velocity net wrapping the frozen photometry model
        self.velocity_net = ResidualVelocityNet(
            phot_velocity_net=phot_model.velocity_net,
            input_dim=1 + property_dim + feature_dim + phot_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim,
            phot_dim=phot_dim,
            feature_dim=feature_dim
        )

    def forward(self, t, x, img, phot):
        """
        t    : (batch, 1)
        x    : (batch, property_dim)
        img  : (batch, 3, H, W)
        phot : (batch, phot_dim)
        """
        img_feat = self.encoder(img)                    # (batch, feature_dim)
        cond_feat = torch.cat([img_feat, phot], dim=1)  # (batch, feature_dim + phot_dim)
        return self.velocity_net(t, x, cond_feat)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class MorphologyEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Initial projection: 152x152 -> 76x76
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Residual Stages
        self.layer1 = ResBlock(64, 64, stride=1)   # 76x76
        self.layer2 = ResBlock(64, 128, stride=2)  # 38x38
        self.layer3 = ResBlock(128, 256, stride=2) # 19x19
        self.layer4 = ResBlock(256, 512, stride=2) # 10x10
        
        # Attention-based pooling instead of simple average
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class ImprovedConditionalFlowModelExperimental(nn.Module):
    """
    Image + Photometry CFM using MorphologyEncoder (ResNet-style with residual
    blocks) and ImprovedVelocityNet (5-layer 512-wide MLP).
    Drop-in replacement for ImprovedConditionalFlowModel with a stronger
    image encoder designed to capture galaxy morphological features.
    """
    def __init__(self,
                 property_dim,
                 feature_dim=256,
                 phot_dim=5,
                 hidden_dim=512):
        super().__init__()
        self.encoder = MorphologyEncoder(feature_dim=feature_dim)
        self.velocity_net = ImprovedVelocityNet(
            input_dim=1 + property_dim + feature_dim + phot_dim,
            hidden_dim=hidden_dim,
            output_dim=property_dim
        )

    def forward(self, t, x, img, phot):
        """
        t:    (batch, 1)
        x:    (batch, property_dim)
        img:  (batch, 3, H, W)
        phot: (batch, phot_dim)
        """
        img_feat = self.encoder(img)
        cond_feat = torch.cat([img_feat, phot], dim=1)
        return self.velocity_net(t, x, cond_feat)
