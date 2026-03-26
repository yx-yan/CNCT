import tigre
import tigre.algorithms as algs
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# 1. 读取 .nii.gz 文件
# ─────────────────────────────────────────────
nii_path = "/projects/_hdd/CTdata/AbdomenCT-1K-ImagePart1/Case_00001_0000.nii.gz"

sitk_img = sitk.ReadImage(nii_path)
volume   = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

# 读取真实物理 spacing
spacing  = sitk_img.GetSpacing()   # (sx, sy, sz) 单位 mm
sx, sy, sz = spacing
print("Spacing (x,y,z) mm:", spacing)
print("Volume shape (Z,Y,X):", volume.shape)

nZ, nY, nX = volume.shape

# HU 值转线性衰减系数
mu_water = 0.02   # mm^-1
volume   = (volume + 1000.0) / 1000.0 * mu_water
volume   = np.clip(volume, 0, None)
volume   = np.ascontiguousarray(volume)

# ─────────────────────────────────────────────
# 2. 定义锥束几何参数
# ─────────────────────────────────────────────
geo = tigre.geometry()
geo.mode = "cone"

# 体积参数
geo.nVoxel = np.array([nZ, nY, nX])
geo.sVoxel = np.array([nZ*sz, nY*sy, nX*sx], dtype=np.float32)
geo.dVoxel = geo.sVoxel / geo.nVoxel

# DSO 必须大于体积对角线半径，保证光源在物体外部
max_radius = np.sqrt((geo.sVoxel[1]/2)**2 + (geo.sVoxel[2]/2)**2)
geo.DSO = max_radius * 5        # ← 从3倍改为5倍，增大有效重建区域，消除黑圆圈
geo.DSD = geo.DSO * 1.5

magnification = geo.DSD / geo.DSO

# 探测器列方向额外放大1.5倍，确保完整覆盖物体并消除截断伪影
geo.nDetector = np.array([nZ, max(nY, nX)])
geo.dDetector = np.array([sz * magnification,
                           sx * magnification * 1.5])   # ← 列方向额外放大1.5倍
geo.sDetector = geo.nDetector * geo.dDetector

geo.offOrigin   = np.array([0, 0, 0])
geo.offDetector = np.array([0, 0])
geo.accuracy    = 0.5

# 验证探测器是否完整覆盖物体
print("\n--- 几何参数验证 ---")
print(f"体积物理尺寸 (Z,Y,X) mm: {geo.sVoxel}")
print(f"探测器物理尺寸 (Z,X) mm: {geo.sDetector}")
print(f"放大率: {magnification:.3f}")
print(f"体积Y方向放大后需覆盖: {geo.sVoxel[1] * magnification:.1f} mm")
print(f"探测器X方向实际尺寸:   {geo.sDetector[1]:.1f} mm")
assert geo.sDetector[0] >= geo.sVoxel[0], "探测器Z方向未覆盖完整体积！"
assert geo.sDetector[1] >= geo.sVoxel[1] * magnification, "探测器X方向未覆盖完整体积！"
print("✅ 探测器覆盖验证通过")
print("--------------------\n")

print(geo)

# ─────────────────────────────────────────────
# 3. 定义投影角度
# ─────────────────────────────────────────────
num_angles = 360
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

# ─────────────────────────────────────────────
# 4. 正向投影：CT → 投影
# ─────────────────────────────────────────────
print("Forward projecting...")
projections = tigre.Ax(volume, geo, angles)
print("Projections shape:", projections.shape)

# ─────────────────────────────────────────────
# 5. 重建：投影 → CT（锥束使用 FDK）
# ─────────────────────────────────────────────
print("Reconstructing with FDK...")
imgFDK = algs.fdk(projections, geo, angles)
print("Reconstruction shape:", imgFDK.shape)

# ─────────────────────────────────────────────
# 6. 创建输出文件夹
# ─────────────────────────────────────────────
os.makedirs("output_projections", exist_ok=True)
os.makedirs("output_recon", exist_ok=True)

# ─────────────────────────────────────────────
# 7. 保存每个角度的投影图（每10个角度保存一张）
# 投影使用全局统一的 vmin/vmax（物理含义相同，应反映真实差异）
# ─────────────────────────────────────────────
print("Saving projection images...")
vmin_proj = projections.min()
vmax_proj = projections.max()

for i in range(0, num_angles, 10):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(projections[i], cmap="gray", aspect="auto",
              vmin=vmin_proj, vmax=vmax_proj)
    ax.set_title(f"Projection {i+1}/{num_angles}  |  Angle: {np.degrees(angles[i]):.1f}°")
    ax.set_xlabel("Detector Column")
    ax.set_ylabel("Z slice")
    plt.tight_layout()
    plt.savefig(f"output_projections/proj_{i:04d}.png", dpi=100)
    plt.close()

print("Projection images saved to output_projections/")

# ─────────────────────────────────────────────
# 8. 保存 Sinogram
# ─────────────────────────────────────────────
mid_z = nZ // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(projections[0, mid_z, :])
axes[0].set_title(f"Single Projection Line\n(Angle 0°, Z slice {mid_z})")
axes[0].set_xlabel("Detector Column")
axes[0].set_ylabel("Intensity")

axes[1].imshow(projections[:, mid_z, :], cmap="gray", aspect="auto")
axes[1].set_title(f"Sinogram (Z slice {mid_z})")
axes[1].set_xlabel("Detector Column")
axes[1].set_ylabel("Angle Index")

plt.tight_layout()
plt.savefig("output_projections/sinogram.png", dpi=150)
plt.close()
print("Sinogram saved to output_projections/sinogram.png")

# ─────────────────────────────────────────────
# 9. 保存重建切片（每5层保存一张）
# 重建切片每张单独用 percentile 设置对比度
# 避免全局 vmax 被高值切片主导导致其他切片偏暗
# ─────────────────────────────────────────────
print("Saving reconstruction slices...")

for z in range(0, nZ, 5):
    fig, ax = plt.subplots(figsize=(6, 6))
    slice_data = imgFDK[z]
    vmin = np.percentile(slice_data, 1)    # 忽略最低1%的极端值
    vmax = np.percentile(slice_data, 99)   # 忽略最高1%的极端值
    ax.imshow(slice_data, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"FDK Reconstructed Slice {z+1}/{nZ}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"output_recon/recon_slice_{z:04d}.png", dpi=100)
    plt.close()

print("Reconstruction slices saved to output_recon/")

# ─────────────────────────────────────────────
# 10. 原始 vs 重建对比总览（均匀采样9张切片）
# 原始和重建各自用 percentile 保证对比度一致
# ─────────────────────────────────────────────
sample_slices = np.linspace(0, nZ - 1, 9, dtype=int)
fig, axes = plt.subplots(2, 9, figsize=(24, 6))
fig.suptitle("Original (top) vs FDK Reconstructed (bottom)", fontsize=14)

for col, z in enumerate(sample_slices):
    orig  = volume[z]
    recon = imgFDK[z]

    axes[0, col].imshow(orig, cmap="gray",
                        vmin=np.percentile(orig,  1),
                        vmax=np.percentile(orig,  99))
    axes[0, col].set_title(f"Slice {z}", fontsize=8)
    axes[0, col].axis("off")

    axes[1, col].imshow(recon, cmap="gray",
                        vmin=np.percentile(recon, 1),
                        vmax=np.percentile(recon, 99))
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig("output_recon/comparison_overview.png", dpi=150)
plt.close()
print("Comparison overview saved to output_recon/comparison_overview.png")

# ─────────────────────────────────────────────
# 11. 保存重建结果为 .nii.gz
# ─────────────────────────────────────────────
recon_sitk = sitk.GetImageFromArray(imgFDK)
recon_sitk.SetSpacing(sitk_img.GetSpacing())
recon_sitk.SetOrigin(sitk_img.GetOrigin())
recon_sitk.SetDirection(sitk_img.GetDirection())
sitk.WriteImage(recon_sitk, "output_recon/reconstructed.nii.gz")
print("Reconstructed NIfTI saved to output_recon/reconstructed.nii.gz")

print("\n✅ All done!")
print("  Projections   → output_projections/  (proj_XXXX.png + sinogram.png)")
print("  Reconstruction → output_recon/       (recon_slice_XXXX.png + comparison_overview.png + reconstructed.nii.gz)")
