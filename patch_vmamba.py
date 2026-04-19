"""
Patch grid_proto_fewshot.py to add VMamba backbone support
"""
import re

filepath = "models/grid_proto_fewshot.py"
with open(filepath, "r") as f:
    code = f.read()

# === PATCH 1: Add vmamba option in get_encoder() ===
# Insert after the dinov2_b14 block, before the 'else: raise'
old_encoder = """        else:
            raise NotImplementedError(
                f'Backbone network {self.config[\"which_model\"]} not implemented')"""

new_encoder = """        elif self.config['which_model'] == 'vmamba_tiny':
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
                self.image_size//32, DEFAULT_FEATURE_SIZE), max(self.image_size//32, DEFAULT_FEATURE_SIZE)]
        else:
            raise NotImplementedError(
                f'Backbone network {self.config[\"which_model\"]} not implemented')"""

assert old_encoder in code, "PATCH 1 FAILED: could not find encoder block to patch"
code = code.replace(old_encoder, new_encoder)
print("PATCH 1 OK: vmamba added to get_encoder()")

# === PATCH 2: Add vmamba option in get_features() ===
old_features = """        else:
            raise NotImplementedError(
                f'Backbone network {self.config[\"which_model\"]} not implemented')
        
        return img_fts"""

new_features = """        elif 'vmamba' in self.config['which_model']:
            # VMamba produces list of multi-scale features, we use the last stage
            outs = self.encoder(imgs_concat)
            img_fts = outs[-1]  # B, C, H, W (already channel-first from Backbone_VSSM)
            if img_fts.shape[-1] < DEFAULT_FEATURE_SIZE:
                img_fts = F.interpolate(img_fts, size=(
                    DEFAULT_FEATURE_SIZE, DEFAULT_FEATURE_SIZE), mode='bilinear')
        else:
            raise NotImplementedError(
                f'Backbone network {self.config[\"which_model\"]} not implemented')
        
        return img_fts"""

assert old_features in code, "PATCH 2 FAILED: could not find features block to patch"
code = code.replace(old_features, new_features)
print("PATCH 2 OK: vmamba added to get_features()")

# === PATCH 3: Add vmamba embed_dim in get_cls() ===
old_cls = """            if 'dinov2_b14' in self.config['which_model']:
                embed_dim = 768
            elif 'dinov2_l14' in self.config['which_model']:
                embed_dim = 1024"""

new_cls = """            if 'dinov2_b14' in self.config['which_model']:
                embed_dim = 768
            elif 'dinov2_l14' in self.config['which_model']:
                embed_dim = 1024
            elif 'vmamba_tiny' in self.config['which_model']:
                embed_dim = 768"""

assert old_cls in code, "PATCH 3 FAILED: could not find cls block to patch"
code = code.replace(old_cls, new_cls)
print("PATCH 3 OK: vmamba embed_dim added to get_cls()")

# Write the patched file
with open(filepath, "w") as f:
    f.write(code)

print("\nAll patches applied successfully!")
