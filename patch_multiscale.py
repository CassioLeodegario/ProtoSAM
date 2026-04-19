"""
Patch grid_proto_fewshot.py to use multi-scale VMamba features
"""
filepath = "models/grid_proto_fewshot.py"
with open(filepath, "r") as f:
    code = f.read()

# === PATCH 1: Change out_indices and add projection layer in get_encoder() ===
old_encoder = """        elif self.config['which_model'] == 'vmamba_tiny':
            self.encoder = Backbone_VSSM(
                out_indices=(3,),
                pretrained='pretrained_model/vmamba_tiny_v2.pth',
                depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2,
                patch_size=4, in_chans=3, num_classes=1000,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
                ssm_init="v0", forward_type="v05_noz",
                mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                patch_norm=True, norm_layer="ln2d",
                downsample_version="v3", patchembed_version="v2",
                use_checkpoint=False, posembed=False, imgsize=224,
            )
            self.config['feature_hw'] = [max(
                self.image_size//32, DEFAULT_FEATURE_SIZE), max(self.image_size//32, DEFAULT_FEATURE_SIZE)]"""

new_encoder = """        elif self.config['which_model'] == 'vmamba_tiny':
            self.encoder = Backbone_VSSM(
                out_indices=(1, 2, 3),
                pretrained='pretrained_model/vmamba_tiny_v2.pth',
                depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2,
                patch_size=4, in_chans=3, num_classes=1000,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
                ssm_init="v0", forward_type="v05_noz",
                mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                patch_norm=True, norm_layer="ln2d",
                downsample_version="v3", patchembed_version="v2",
                use_checkpoint=False, posembed=False, imgsize=224,
            )
            # Multi-scale projection: stages 1,2,3 have dims 192, 384, 768
            # Concatenated = 192+384+768 = 1344, project to 768
            self.vmamba_proj = nn.Conv2d(1344, 768, kernel_size=1, bias=False)
            self.config['feature_hw'] = [max(
                self.image_size//32, DEFAULT_FEATURE_SIZE), max(self.image_size//32, DEFAULT_FEATURE_SIZE)]"""

assert old_encoder in code, "PATCH 1 FAILED: could not find encoder block"
code = code.replace(old_encoder, new_encoder)
print("PATCH 1 OK: encoder changed to multi-scale")

# === PATCH 2: Change get_features() for multi-scale ===
old_features = """        elif 'vmamba' in self.config['which_model']:
            # VMamba produces list of multi-scale features, we use the last stage
            outs = self.encoder(imgs_concat)
            img_fts = outs[-1]  # B, C, H, W (already channel-first from Backbone_VSSM)
            if img_fts.shape[-1] < DEFAULT_FEATURE_SIZE:
                img_fts = F.interpolate(img_fts, size=(
                    DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE), mode='bilinear')"""

new_features = """        elif 'vmamba' in self.config['which_model']:
            # VMamba multi-scale: upsample stages 1,2,3 to same size, concat, project
            outs = self.encoder(imgs_concat)
            target_size = outs[-1].shape[-2:]  # use last stage resolution
            upsampled = []
            for feat in outs:
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                upsampled.append(feat)
            img_fts = torch.cat(upsampled, dim=1)  # B, 1344, H, W
            img_fts = self.vmamba_proj(img_fts)  # B, 768, H, W
            if img_fts.shape[-1] < DEFAULT_FEATURE_SIZE:
                img_fts = F.interpolate(img_fts, size=(
                    DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE), mode='bilinear')"""

assert old_features in code, "PATCH 2 FAILED: could not find features block"
code = code.replace(old_features, new_features)
print("PATCH 2 OK: features changed to multi-scale")

with open(filepath, "w") as f:
    f.write(code)

print("\nAll patches applied successfully!")
