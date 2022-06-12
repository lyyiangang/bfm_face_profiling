import torch
import pdb

def read_txt(txtfile):
    with open(txtfile, "r") as f:
        lines = f.readlines()
    lines = [int(l.strip("\n")) for l in lines]
    lines = torch.LongTensor(lines)
    return lines

idx_eyes1 = read_txt("./photometric_optimization/data/verts_eye1.txt")
idx_eyes2 = read_txt("./photometric_optimization/data/verts_eye2.txt")
idx_mouth = read_txt("./photometric_optimization/data/verts_mouth.txt")
idx_bd = read_txt("./photometric_optimization/data/verts_bd.txt")
idx_nosetip = 3098

def get_full_verts(verts):
    # add eyes
    verts_eyes1 = verts[:,idx_eyes1]
    verts_eyes2 = verts[:,idx_eyes2]
    # add mouth
    verts_mouth = verts[:,idx_mouth]
    mouth_center = verts_mouth.mean(1, keepdims=True)
    list_mouth_append = [verts_mouth]
    for _r, _c in zip([0.5, 0.2], [1, 0.9]):
        mouth_center_in = mouth_center.clone()
        mouth_center_in[:,:,2] = verts_mouth[:,:, 2].min(1, keepdims=True)[0] * _c
        verts_mouth2 = verts_mouth * _r
        mc = verts_mouth2.mean(1, keepdims=True)
        verts_mouth2 = verts_mouth2 - (mc - mouth_center_in)
        list_mouth_append.append(verts_mouth2)
    mc = verts_mouth2.mean(1, keepdims=True)
    mc_z = verts_mouth2[:, :,2].min(1)[0][..., None]
    mc[:,:,2] = mc_z
    list_mouth_append.append(mc)
    verts_mouth = torch.cat(list_mouth_append, 1)

    zmean = verts[:,:,2].min(1, keepdims=True)[0]

    verts_bd = verts[:,idx_bd].clone()
    list_bd_append = [verts_bd]
    interval = (zmean-verts_bd[:,:,2])/3
    bdmean = verts_bd[:,:,2].min(1, keepdims=True)[0]
    ds = [bdmean+interval*r for r in [0.5]]
    for rate,d in zip([1.4], ds):
        center = verts_bd.mean(1, keepdims=True)
        verts_bd = verts_bd-center
        verts_bd2 = verts_bd * rate+center
        verts_bd2[:,:,2] = d
        #verts_bd2[:, :2] -= verts[idx_nosetip][:2] * rate
        #verts_bd2[:, 2] -= (verts[idx_nosetip][2] * (rate - tr))
        list_bd_append.append(verts_bd2)
        verts_bd = verts_bd2
    verts_bd = torch.cat(list_bd_append, 1)
    N_bd = verts_bd.size(1)

    verts_full = torch.cat((verts, verts_eyes1, verts_eyes2, verts_mouth, verts_bd), 1)
    return verts_full, N_bd

def image_meshing(verts, N_bd):
    zmean = verts[:,-N_bd:,2].max(1, keepdims=True)[0]
    xval = torch.linspace(-1, 1, 10).view(1,-1).expand(verts.size(0),-1)
    yval = -torch.linspace(-1, 1, 10).view(1,-1).expand(verts.size(0),-1)

    leftbd = torch.stack((torch.ones_like(yval)*xval[0, 0], yval, torch.ones_like(yval)*zmean), 2)
    rightbd = torch.stack((torch.ones_like(yval)*xval[0,-1], yval, torch.ones_like(yval)*zmean), 2)
    topbd = torch.stack((xval, torch.ones_like(xval)*yval[0,-1], torch.ones_like(xval)*zmean),2)
    btmbd = torch.stack((xval, torch.ones_like(xval)*yval[0,0], torch.ones_like(xval)*zmean),2)

    list_bd_image = [leftbd, rightbd, topbd, btmbd]
    verts_bd = torch.cat(list_bd_image, 1)
    verts_full = torch.cat((verts, verts_bd),1)
    return verts_full
