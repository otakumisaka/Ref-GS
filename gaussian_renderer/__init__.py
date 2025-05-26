import copy
import math
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F

from diff_surfel_2dgs import GaussianRasterizationSettings as GaussianRasterizationSettings_2dgs
from diff_surfel_2dgs import GaussianRasterizer as GaussianRasterizer_2dgs
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization_real import GaussianRasterizationSettings as GaussianRasterizationSettings_real
from diff_surfel_rasterization_real import GaussianRasterizer as GaussianRasterizer_real

from scene.gaussian_model import GaussianModel

from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.graphics_utils import fov2focal

from utils.color_utils import *
from utils.sph_utils import *

# Debug
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'


DIR="result/"
use_feature = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

def get_inside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) <= ENV_RADIUS**2


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, iteration=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings_black = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=use_feature,
    )
    
    rasterizer_black = GaussianRasterizer(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    rets =  {}

    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness
    gs_feature = pc.get_language_feature
    gs_ior = pc.get_ior
    
    # print(f"pc.gs_ior: {gs_ior}")

    # roughness and feature set in Model? Every point refers to a roughness/feature value
    # we view ior as one of its features
    input_ts = torch.cat([gs_roughness, gs_ior, gs_feature], dim=-1)
    print(f"pc.input_ts: {input_ts.shape}, pc.gsfeat_dim: {pc.gsfeat_dim}")
    print(f"gs_roughness: {gs_roughness.shape}, gs_feature: {gs_feature.shape}, gs_ior: {gs_ior.shape}")
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = gs_albedo,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    # to world coordinate
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    #####################################################################################################################
    # reflection 
    viewdirs = viewpoint_camera.rays_d
    normals = render_normal.permute(1,2,0)
    wo = F.normalize(reflect(-viewdirs, normals), dim=-1)
    
    #refraction
    
    out_ts = out_ts.permute(1,2,0)

    albedo_map = albedo_map.permute(1,2,0)
    roughness_map = out_ts[..., 0:1]
    ior_map = out_ts[..., 1:2]
    feature_map = out_ts[..., 2:6]
    print(f"pc.feature_map: {feature_map.shape}, pc.ior_map: {ior_map.shape}, pc.albedo_map: {albedo_map.shape}, pc.roughness_map: {roughness_map.shape}")

    #####################################################################################################################
    
    with torch.no_grad():
        select_index = (render_alpha.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
    
    wo = wo.reshape(-1, 3)[select_index]
    normals = normals.reshape(-1, 3)[select_index]
    roughness_map = roughness_map.reshape(-1, 1)[select_index]
    albedo_map = albedo_map.reshape(-1, 3)[select_index]
    ior_map = ior_map.reshape(-1, 1)[select_index]
    
    feature_map = feature_map.reshape(-1, pc.gsfeat_dim)[select_index]
    feature_map = F.normalize(feature_map, dim=-1)
    
    feature_map = feature_map.reshape(-1, 1, pc.gsfeat_dim)
    feature_dirc = feature_map.reshape(-1, pc.gsfeat_dim)
    
    print(f"feature_map.shape:{feature_map.shape}, feature_dirc.shape: {feature_dirc.shape}")

    ''' Sph-Mip '''
    wo_xy = (cart2sph(wo.reshape(-1, 3)[..., [0,1,2]])[..., 1:] / torch.Tensor([[np.pi, 2*np.pi]]).to(device))[..., [1,0]]
    wo_xyz = torch.stack([wo_xy[:, None, :]], dim=0,)
    
    spec_level = roughness_map.reshape(-1, 1)

    spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
    spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
    spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)
    
    #####################################################################################################################
    
        # 计算入射方向和折射方向
    incident_dir = viewdirs.reshape(-1, 3)[select_index]  # 从相机到表面的方向
    surface_normals = normals  # 表面法线
    
    # 折射率处理：限制在合理范围内，避免极端值
    ior_values = torch.clamp(ior_map, min=1.0, max=3.0)
    
    # from incident to exitant
    # eta1 / eta2
    eta = 1.0 / ior_values
    
    wt, total_reflection = refract(incident_dir, surface_normals, eta)
    
    cos_theta_i = torch.clamp(-torch.sum(incident_dir * surface_normals, dim=-1, keepdim=True), 0.0, 1.0)
    
    # Fresnel ratio 
    f0 = ((ior_values - 1.0) / (ior_values + 1.0)) ** 2  # 垂直入射时的反射率
    fresnel = fresnel_schlick(cos_theta_i, f0)
    
    # Sph-Mip for refraction
    # we know wt must be non-zero if it is not total internal reflection
    valid_refraction = ~total_reflection.squeeze(-1)
    
    if valid_refraction.sum() > 0:
        wt_valid = wt[valid_refraction]
        wt_xy = (cart2sph(wt_valid.reshape(-1, 3)[..., [0,1,2]])[..., 1:] / 
                 torch.Tensor([[np.pi, 2*np.pi]]).to(device))[..., [1,0]]
        wt_xyz = torch.stack([wt_xy[:, None, :]], dim=0)
        
        # like spec_level
        refract_level = roughness_map[valid_refraction].reshape(-1, 1)
        refract_feat = pc.dir_encoding(wt_xyz, refract_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
        
        refract_feat_wrap = refract_feat.reshape(-1, pc.sph_dim, 1)
        refract_feat_dirc = refract_feat.reshape(-1, pc.sph_dim)
        
        # 折射光照计算
        feature_map_valid = feature_map[valid_refraction]
        refract_wrap_input = (refract_feat_wrap @ feature_map_valid).reshape(-1, pc.sph_dim * pc.gsfeat_dim)
        refract_input_mlp = torch.cat([refract_wrap_input, refract_feat_dirc], -1)
        
        # 使用相同的MLP或可以定义专门的折射MLP
        refract_mlp_output = pc.light_mlp(refract_input_mlp).float()
        refract_light_valid = torch.exp(torch.clamp(refract_mlp_output, max=5.0))
        
        # 创建完整的折射光数组
        refract_light = torch.zeros_like(spec_light)
        refract_light[valid_refraction] = refract_light_valid
    else:
        refract_light = torch.zeros_like(spec_light)
        
    # Specular color
    wrap_input = (spec_feat_wrap @ feature_map).reshape(-1, pc.sph_dim*pc.gsfeat_dim)
    input_mlp = torch.cat([wrap_input, spec_feat_dirc], -1)
    mlp_output = pc.light_mlp(input_mlp).float()
    spec_light = torch.exp(torch.clamp(mlp_output, max=5.0))
    
    # Diffuse color
    diff_light = albedo_map
    
    # Refraction color
    transparency = torch.clamp((ior_values - 1.0) / 2.0, 0.0, 1.0)  
    
    reflection_contrib = fresnel * spec_light
    refraction_contrib = (1.0 - fresnel) * transparency * refract_light
    diffuse_contrib = (1.0 - transparency) * diff_light
    
    # pbr_rgb = spec_light + diff_light   
    pbr_rgb = reflection_contrib + refraction_contrib + diffuse_contrib     
    pbr_rgb = linear2srgb(pbr_rgb)
    pbr_rgb = torch.clamp(pbr_rgb, min=0., max=1.)
    
    #####################################################################################################################
    
    output_rgb = torch.zeros(image_height, image_width, 3).to(device)
    output_rgb.reshape(-1, 3)[select_index] = pbr_rgb
    output_rgb = output_rgb.permute(2,0,1)
    
    rets.update({
        'pbr_rgb': output_rgb,

        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 
        
    if iteration % 500 == 0:
        with torch.no_grad():
            
            output_spec = torch.zeros(image_height, image_width, 3).to(device)
            output_spec.reshape(-1, 3)[select_index] = linear2srgb(spec_light)
            output_spec = output_spec.permute(2,0,1)
            
            output_diff = torch.zeros(image_height, image_width, 3).to(device)
            output_diff.reshape(-1, 3)[select_index] = linear2srgb(diff_light)
            output_diff = output_diff.permute(2,0,1)
    
            torchvision.utils.save_image(render_alpha, DIR+"render_alpha.png")
            torchvision.utils.save_image(((render_normal+1)/2)*render_alpha, DIR+"render_normal.png")
            
            surf_depth = surf_depth / surf_depth.max()
            torchvision.utils.save_image(surf_depth, DIR+"surf_depth.png")
            torchvision.utils.save_image((surf_normal+1)/2, DIR+"surf_normal.png")
            
            gt_image_copy = viewpoint_camera.original_image.to(device)
            torchvision.utils.save_image(gt_image_copy, DIR+"gt_image.png")
            
            torchvision.utils.save_image(((out_ts[..., 1:].permute(2,0,1))[:3]+1)/2, DIR+"feature_image.png")
            torchvision.utils.save_image(out_ts[..., :1].repeat(1,1,3).permute(2,0,1), DIR+"roughness.png")
            
            torchvision.utils.save_image(output_rgb*render_alpha, DIR+"pbr_rgb.png")
            torchvision.utils.save_image(output_spec, DIR+"spec_light.png")
            torchvision.utils.save_image(output_diff, DIR+"diff_light.png")

    
    return rets

# TODO: transmittance not updated now, I'm not sure whether this part ought to be updated
def render_nerf(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, iteration=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings_2dgs(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    raster_settings_black = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=use_feature,
    )
    
    rasterizer = GaussianRasterizer_2dgs(raster_settings=raster_settings)
    rasterizer_black = GaussianRasterizer(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    rendered_image, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    rets =  {
        "render": rendered_image,
    }

    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness
    gs_feature = pc.get_language_feature
    
    input_ts = torch.cat([gs_roughness, gs_feature], dim=-1)
    
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = gs_albedo,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    #####################################################################################################################
    
    viewdirs = viewpoint_camera.rays_d
    normals = render_normal.permute(1,2,0)
    wo = F.normalize(reflect(-viewdirs, normals), dim=-1)
    
    out_ts = out_ts.permute(1,2,0)
    
    albedo_map = albedo_map.permute(1,2,0)
    roughness_map = out_ts[..., :1]
    feature_map = out_ts[..., 1:]
    
    #####################################################################################################################
    
    with torch.no_grad():
        select_index = (render_alpha.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
    
    wo = wo.reshape(-1, 3)[select_index]
    normals = normals.reshape(-1, 3)[select_index]
    roughness_map = roughness_map.reshape(-1, 1)[select_index]
    albedo_map = albedo_map.reshape(-1, 3)[select_index]
    
    feature_map = feature_map.reshape(-1, pc.gsfeat_dim)[select_index]
    feature_map = F.normalize(feature_map, dim=-1)
    
    feature_map = feature_map.reshape(-1, 1, pc.gsfeat_dim)
    feature_dirc = feature_map.reshape(-1, pc.gsfeat_dim)
    
    ''' Sph-Mip '''
    wo_xy = (cart2sph(wo.reshape(-1, 3)[..., [0,1,2]])[..., 1:] / torch.Tensor([[np.pi, 2*np.pi]]).to(device))[..., [1,0]]
    wo_xyz = torch.stack([wo_xy[:, None, :]], dim=0,)
    
    spec_level = roughness_map.reshape(-1, 1)

    spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
    spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
    spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)
    
    #####################################################################################################################
    
    # Specular color
    wrap_input = (spec_feat_wrap @ feature_map).reshape(-1, pc.sph_dim*pc.gsfeat_dim)
    input_mlp = torch.cat([wrap_input, spec_feat_dirc], -1)
    mlp_output = pc.light_mlp(input_mlp).float()
    spec_light = torch.exp(torch.clamp(mlp_output, max=5.0))
    
    # Diffuse color
    diff_light = albedo_map
    
    pbr_rgb = spec_light + diff_light        
    pbr_rgb = linear2srgb(pbr_rgb)
    pbr_rgb = torch.clamp(pbr_rgb, min=0., max=1.)
    
    #####################################################################################################################
    
    output_rgb = torch.zeros(image_height, image_width, 3).to(device)
    output_rgb.reshape(-1, 3)[select_index] = pbr_rgb
    output_rgb = output_rgb.permute(2,0,1)
    
    rets.update({
        'pbr_rgb': output_rgb,

        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 
        
    if iteration % 500 == 0:
        with torch.no_grad():
            torchvision.utils.save_image(rendered_image, DIR+"rendered_image.png")
            
            output_spec = torch.zeros(image_height, image_width, 3).to(device)
            output_spec.reshape(-1, 3)[select_index] = linear2srgb(spec_light)
            output_spec = output_spec.permute(2,0,1)
            
            output_diff = torch.zeros(image_height, image_width, 3).to(device)
            output_diff.reshape(-1, 3)[select_index] = linear2srgb(diff_light)
            output_diff = output_diff.permute(2,0,1)
    
            torchvision.utils.save_image(render_alpha, DIR+"render_alpha.png")
            torchvision.utils.save_image(((render_normal+1)/2)*render_alpha, DIR+"render_normal.png")
            
            surf_depth = surf_depth / surf_depth.max()
            torchvision.utils.save_image(surf_depth, DIR+"surf_depth.png")
            torchvision.utils.save_image((surf_normal+1)/2, DIR+"surf_normal.png")
            
            gt_image_copy = viewpoint_camera.original_image.to(device)
            torchvision.utils.save_image(gt_image_copy, DIR+"gt_image.png")
            
            torchvision.utils.save_image(((out_ts[..., 1:].permute(2,0,1))[:3]+1)/2, DIR+"feature_image.png")
            torchvision.utils.save_image(out_ts[..., :1].repeat(1,1,3).permute(2,0,1), DIR+"roughness.png")
            
            torchvision.utils.save_image(output_rgb*render_alpha, DIR+"pbr_rgb.png")
            torchvision.utils.save_image(output_spec, DIR+"spec_light.png")
            torchvision.utils.save_image(output_diff, DIR+"diff_light.png")
            
    return rets

def render_real(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, iteration=0, ITER=0, ENV_CENTER=None, ENV_RADIUS=None, XYZ=[0,1,2]):
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings_2dgs(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    raster_settings_black = GaussianRasterizationSettings_real(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        include_feature=use_feature,
    )
    
    rasterizer = GaussianRasterizer_2dgs(raster_settings=raster_settings)
    rasterizer_black = GaussianRasterizer_real(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    rendered_image, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    rets =  {
        "render": rendered_image,
    }

    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness
    gs_mask = pc.get_mask
    gs_feature = pc.get_language_feature
    
    gs_in = gs_mask * 0 + 1.0
    gs_in[get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS)] = 0.0
    
    gs_out = gs_mask * 0 + 0.0
    gs_out[get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS)] = 1.0

    input_ts = torch.cat([gs_roughness, gs_feature, gs_in, gs_out], dim=-1)
    
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = gs_albedo,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )

    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / (render_alpha))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    #####################################################################################################################
    
    viewdirs = viewpoint_camera.rays_d
    normals = render_normal.permute(1,2,0)
    wo = F.normalize(reflect(-viewdirs, normals), dim=-1)
    
    out_ts = out_ts.permute(1,2,0)
    
    albedo_map = albedo_map.permute(1,2,0)
    roughness_map = out_ts[..., :1]
    feature_map = out_ts[..., 1:5]
    in_map = out_ts[..., 5:6]
    
    #####################################################################################################################
    
    with torch.no_grad():
        select_index = (in_map.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
    
    if len(select_index) > 0:
    
        wo = wo.reshape(-1, 3)[select_index]
        normals = normals.reshape(-1, 3)[select_index]
        roughness_map = roughness_map.reshape(-1, 1)[select_index]
        albedo_map = albedo_map.reshape(-1, 3)[select_index]

        feature_map = feature_map.reshape(-1, pc.gsfeat_dim)[select_index]
        feature_map = F.normalize(feature_map, dim=-1)

        feature_map = feature_map.reshape(-1, 1, pc.gsfeat_dim)
        feature_dirc = feature_map.reshape(-1, pc.gsfeat_dim)

        ''' Specular env. feature '''
        wo_xy = (cart2sph(wo.reshape(-1, 3)[..., XYZ])[..., 1:] / torch.Tensor([[np.pi, 2*np.pi]]).to(device))[..., [1,0]] 

        wo_xyz = torch.stack([wo_xy[:, None, :]], dim=0,)

        spec_level = roughness_map.reshape(-1, 1)

        spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)
        spec_feat_wrap = spec_feat.reshape(-1, pc.sph_dim, 1)
        spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)

        #####################################################################################################################

        # Specular color
        wrap_input = (spec_feat_wrap @ feature_map).reshape(-1, pc.sph_dim*pc.gsfeat_dim)
        input_mlp = torch.cat([wrap_input, spec_feat_dirc,], -1)
        mlp_output = pc.light_mlp(input_mlp).float()
        spec_light = torch.exp(torch.clamp(mlp_output, max=5.0))

        # Diffuse color
        diff_light = albedo_map

        pbr_rgb = spec_light + diff_light
        pbr_rgb = linear2srgb(pbr_rgb)
        pbr_rgb = torch.clamp(pbr_rgb, min=0., max=1.)

        #####################################################################################################################

        output_rgb = torch.zeros(image_height, image_width, 3).to(device)
        output_rgb.reshape(-1, 3)[select_index] = pbr_rgb
        output_rgb = output_rgb.permute(2,0,1)

        ref_w = out_ts[..., 5:6].permute(2,0,1).detach()
        out_w = out_ts[..., 6:7].permute(2,0,1).detach()
        full_rgb = ref_w*output_rgb + out_w*rendered_image
        
    else:
        full_rgb = rendered_image
        ref_w = out_ts[..., 5:6].permute(2,0,1).detach()
        out_w = out_ts[..., 6:7].permute(2,0,1).detach()
    
    rets.update({
        'pbr_rgb': full_rgb,
        'ref_w': ref_w,
        'out_w': out_w,
        'ref_index': get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS),

        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 
        
    if iteration % 500 == 0:
        with torch.no_grad():    
            torchvision.utils.save_image(rendered_image, DIR+"rendered_image.png")
            torchvision.utils.save_image(render_alpha, DIR+"render_alpha.png")
            torchvision.utils.save_image(((render_normal+1)/2)*render_alpha, DIR+"render_normal.png")
            
            surf_depth = surf_depth / surf_depth.max()
            torchvision.utils.save_image(surf_depth, DIR+"surf_depth.png")
            torchvision.utils.save_image((surf_normal+1)/2, DIR+"surf_normal.png")
            
            gt_image_copy = viewpoint_camera.original_image.to(device)
            torchvision.utils.save_image(gt_image_copy, DIR+"gt_image.png")
            
            torchvision.utils.save_image(((out_ts[..., 1:].permute(2,0,1))[:3]+1)/2, DIR+"feature_image.png")

            torchvision.utils.save_image(out_ts[..., :1].repeat(1,1,3).permute(2,0,1), DIR+"roughness.png")
            torchvision.utils.save_image(out_ts[..., 5:6].repeat(1,1,3).permute(2,0,1), DIR+"in.png")
            torchvision.utils.save_image(out_ts[..., 6:7].repeat(1,1,3).permute(2,0,1), DIR+"out.png")

            if len(select_index) > 0:
                output_spec = torch.zeros(image_height, image_width, 3).to(device)
                output_spec.reshape(-1, 3)[select_index] = linear2srgb(spec_light)
                output_spec = output_spec.permute(2,0,1)

                output_diff = torch.zeros(image_height, image_width, 3).to(device)
                output_diff.reshape(-1, 3)[select_index] = linear2srgb(diff_light)
                output_diff = output_diff.permute(2,0,1)

                torchvision.utils.save_image(output_rgb*render_alpha, DIR+"pbr_rgb.png")
                torchvision.utils.save_image(full_rgb, DIR+"full_rgb.png")
                torchvision.utils.save_image(output_spec, DIR+"spec_light.png")
                torchvision.utils.save_image(output_diff, DIR+"diff_light.png")

    return rets
