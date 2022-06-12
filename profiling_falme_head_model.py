import cv2
from get_full_verts_v2 import get_full_verts, image_meshing
import sys
sys.path.append('./photometric_optimization')
from renderer import Pytorch3dRasterizer
import util
import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms
from pytorch3d.io import load_obj
import torchvision
import torch.nn.functional as F

class ImageRenderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(ImageRenderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        verts, faces, _ = load_obj(obj_filename)
        print(f'obj verts.shape:{verts.shape}')
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)

        # faces
        self.register_buffer('faces', faces)

    def forward(self, cam, head_vertices, images, cam_new):
        # project vertices on old cam cs to get uvcoords, 从原图获取每个顶点的纹理信息
        full_vertices, N_bd = get_full_verts(head_vertices) # N_bd:110
        t_vertices = util.batch_orth_proj(full_vertices, cam)
        
        # to pixel coord system
        t_vertices[..., 1:] = -t_vertices[..., 1:]
        t_vertices[...,2] = t_vertices[...,2]+10
        t_vertices = image_meshing(t_vertices, N_bd)
        t_vertices[...,:2] = torch.clamp(t_vertices[...,:2], -1,1)
        t_vertices[:,:,2] =t_vertices[:,:,2]-9 # t_vertices.shape: [1, 3782, 3]
        batch_size = head_vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        uvcoords = t_vertices.clone()
        # Attributes
        uvcoords = torch.cat([uvcoords[:,:,:2], uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        face_vertices = util.face_vertices(uvcoords, self.faces.expand(batch_size, -1, -1))#self.faces:(1, 7322, 3, 3)

        # render on new cam, 旋转人头,再通过纹理信息将人头光栅化
        attributes = face_vertices.detach()
        full_vertices, N_bd = get_full_verts(head_vertices) # add face contour boundary
        transformed_vertices = util.batch_orth_proj(full_vertices, cam_new) # rot head
        transformed_vertices[..., 1:] = - transformed_vertices[..., 1:]
        transformed_vertices[...,2] = transformed_vertices[...,2]+10 

        transformed_vertices = image_meshing(transformed_vertices, N_bd) # add img boundaries

        transformed_vertices[...,:2] = torch.clamp(transformed_vertices[...,:2], -1,1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        results = F.grid_sample(images, grid, align_corners=False)
        return {'rotate_images':results}

if __name__ == "__main__":
    
    test_annot = "outputs/fitted_flame/AFW_AFW_18489332_7_7_4.jpg.obj.npy"
    test_img ="data/AFW_AFW_18489332_7_7_4.jpg" 
    image_size = 120

    param = np.load(test_annot, allow_pickle=True)[()] # come from flame.py, param.shape:
    head_vertices = torch.Tensor(param['verts0'])
    print(f'npy.vertices.shape:{head_vertices.shape}') # [1, 3487, 3])
    cam = torch.Tensor([[-0.01, -0.01, 0, 1, 0, 0.0]])
    # original pos is in pixel coord system, we need convert it to camera coordinate system
    head_vertices[:, :, 0] -= image_size / 2
    head_vertices[:, :, 1] += image_size / 2
    head_vertices[:, :, :] /= 60

    R = pytorch3d.transforms.euler_angles_to_matrix(cam[:,:3], "XYZ")
    print(f'cam:{cam}, R:{R}')
    images = []
    image = cv2.resize(cv2.imread(test_img), (image_size, image_size)).astype(np.float32) / 255.
    image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
    images.append(torch.from_numpy(image[None, :, :, :]))
    images = torch.cat(images, dim=0)

    mesh_file = './photometric_optimization/data/full.obj' # [3782, 3] # only use the facet data
    render = ImageRenderer(image_size, obj_filename=mesh_file)

    angles = torch.Tensor([20, 20, 0])[None,...]/180.0 * np.pi # rotation angles xyz
    cam_new = cam.clone()
    angles = torch.abs(angles)*torch.sign(cam_new[:,:3])
    print(f'rot angles:{angles}')
    cam_new[:,:3] = cam_new[:,:3]+angles
    print(f'cam_new:{cam_new}')

    ops = render(cam, head_vertices, images, cam_new)

    grids = {}
    visind = range(1)  # [0]
    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
    grids['rotateimage'] = torchvision.utils.make_grid(
        (ops['rotate_images'])[visind].detach().cpu())
    grid = torch.cat(list(grids.values()), 1)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

    out_file = 'aug_results/result.jpg'
    print(f'writing {out_file}')
    cv2.imwrite(out_file, grid_image)
