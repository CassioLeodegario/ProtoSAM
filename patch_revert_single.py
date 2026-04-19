filepath = "models/grid_proto_fewshot.py"
with open(filepath, "r") as f:
    code = f.read()

# PATCH 1: Revert encoder to single-stage
old = """            self.encoder = Backbone_VSSM(
                out_indices=(1, 2, 3),"""
new = """            self.encoder = Backbone_VSSM(
                out_indices=(3,),"""
assert old in code, "PATCH 1 FAILED"
code = code.replace(old, new)
print("PATCH 1 OK: out_indices reverted to (3,)")

# PATCH 2: Remove multi-scale projection layer
old = """            )
            # Multi-scale projection: stages 1,2,3 have dims 192, 384, 768
            # Concatenated = 192+384+768 = 1344, project to 768
            self.vmamba_proj = nn.Conv2d(1344, 768, kernel_size=1, bias=False)
            self.config['feature_hw']"""
new = """            )
            self.config['feature_hw']"""
assert old in code, "PATCH 2 FAILED"
code = code.replace(old, new)
print("PATCH 2 OK: projection layer removed")

# PATCH 3: Revert get_features to single-stage
old = """        elif 'vmamba' in self.config['which_model']:
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
            if img_fts.shape[-1] < DEFAULT_FEATURE_SIZE:"""
new = """        elif 'vmamba' in self.config['which_model']:
            # VMamba single-stage: use last stage features only
            outs = self.encoder(imgs_concat)
            img_fts = outs[-1]  # B, C, H, W
            if img_fts.shape[-1] < DEFAULT_FEATURE_SIZE:"""
assert old in code, "PATCH 3 FAILED"
code = code.replace(old, new)
print("PATCH 3 OK: get_features reverted to single-stage")

with open(filepath, "w") as f:
    f.write(code)
print("\nAll patches applied!")
