import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import matplotlib.colors as mcolors


def compose_image_w_gradient_color(C,H,W):
    assert C == 3
    xx = torch.arange(0, W).view(1, -1).repeat(H,1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1,W)
    img = (xx + yy).to(dtype=torch.float)
    img = img / torch.max(img)
    img = img.view(1, H, W).repeat(3, 1, 1)
    return img


def load_img_as_tensor(im_path='/Users/Shasha/Pictures/tangent.png', is_show=False):
    img = mpimg.imread(im_path)
    img = img[:,:,:3]
    # print(type(img), np.shape(img))
    if is_show:
        plt.imshow(img)
        plt.show()
    img = torch.tensor(img).permute(2,0,1)
    return img



def show_img_from_tensor(img):
    img = img.permute([1,2,0]).numpy()
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # C, H, W = 3, 112, 112
    # img = compose_image_w_gradient_color(C,H,W)
    img = load_img_as_tensor(is_show=False)
    theta = torch.tensor([[0.5, 0, 0],
                          [0, 0.5, 0]], dtype=torch.float)

    # what is the range of theta
    align_corners = True
    img_batch = img.unsqueeze(dim=0)
    theta_bath = theta.unsqueeze(dim=0)
    flow_grid = nn.functional.affine_grid(theta_bath, size=img_batch.size(), align_corners=align_corners)
    print(flow_grid.size(), torch.min(flow_grid), torch.max(flow_grid))
    out_img = nn.functional.grid_sample(img_batch, flow_grid, align_corners=align_corners)
    show_img_from_tensor(out_img[0])

