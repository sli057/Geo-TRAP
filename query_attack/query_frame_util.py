import torch
import torch.nn.functional as F
from mmaction.models.geotransfrom.regression_head import PerspectiveRegressionHead
import random

#random.seed(1)
def wrap_with_optical_flow(x, optical_flow, padding_mode='zeros'):
    """Warp an image or feature map with optical flow
        Args:
            x (Tensor): size (n, c, h, w)
            flow (Tensor): size (n, 2,  t,  h, w), values range from -1 to 1 (relevant to image width or height)
            padding_mode (str): 'zeros' or 'border'

        Returns:
            Tensor: warped frames
        """
    out_frames = []
    n, _, t, h, w = list(optical_flow.size())
    x_ = torch.arange(w).view(1, -1).expand(h, -1) #[h,w]
    y_ = torch.arange(h).view(-1, 1).expand(-1, w) #[h,w]
    grid = torch.stack([x_, y_], dim=0).float().to(x.device) #[2, h, w]
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1) #[n, 2, h, w]
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

    last_frames = x
    for tt in range(t):
        flow = optical_flow[:, :, tt, :, :] # [n, 2, H, W]
        # print(grid.size(), flow.size())
        cur_grid = grid + 2 * flow
        cur_grid = cur_grid.permute(0, 2, 3, 1) #[n,h,w,2]
        cur_frames = F.grid_sample(last_frames, cur_grid, padding_mode=padding_mode, align_corners=True)
        out_frames.append(cur_frames.unsqueeze(-3)) # [n, c, 1, h, w]
        last_frames = cur_frames
    out_frames = torch.cat(out_frames, dim=-3) # [n, c, t, h, w]
    return out_frames


def norm2(x):
    x = x.squeeze(dim=0)
    assert len(list(x.size())) == 4 #[c, t, h, w]
    x = x.permute(1, 0, 2, 3) # [t, c, h, w]
    norm_vec = torch.sqrt(x.float().pow(2).sum(dim=[1,2,3])).view(1, 1, -1, 1, 1) # [b, c, t, h, w]
    norm_vec += (norm_vec == 0).float() * 1e-8
    # print(norm_vec.squeeze())
    return norm_vec

def untargeted_pert_loss(logits, label, margin=0.05):
    """Args:
        logits (torch.Tensor): logits.
        label (torch.Tensor): The ground truth label.
        kwargs: Any keyword argument to be used to calculate
            CrossEntropy loss.

    Returns:
        torch.Tensor: The returned CrossEntropy loss.
    """
    # cls_score #[N,C]
    # label N
    #loss_cls = F.cross_entropy(logits, label)
    logits = logits.to(label.device)
    softmax = F.softmax(logits)
    # print(softmax)
    N, C = list(softmax.size())
    one_hot_label = F.one_hot(label, C)#.to(logits.device)
    # print(one_hot_label)
    label_prob = torch.masked_select(softmax, one_hot_label.bool())
    max_non_label_info = torch.max(softmax-one_hot_label, dim=1)
    max_non_label_prob, max_non_label_cls = max_non_label_info.values, max_non_label_info.indices
    loss_margin = margin
    # print(softmax)
    # print(label)
    # print(label_prob)
    # print(max_non_label_prob)

    l_1 = torch.tensor([0.0]*N).to(softmax.device)
    lm = label_prob - max_non_label_prob + loss_margin
    l_2 = lm**2 / loss_margin
    l_3 = lm
    adv_loss = torch.mean(torch.max(l_1, torch.min(l_2, l_3)))
    return adv_loss, label_prob, max_non_label_prob, max_non_label_cls

def untargeted_cross_entropy_pert_loss(logits, label, margin=0.05):
    """Args:
        logits (torch.Tensor): logits.
        label (torch.Tensor): The target label.
        kwargs: Any keyword argument to be used to calculate
            CrossEntropy loss.

    Returns:
        torch.Tensor: The returned CrossEntropy loss.
    """
    # cls_score #[N,C]
    # label N
    logits = logits.to(label.device)
    loss_cls = F.cross_entropy(1.0-logits, label)
    softmax = F.softmax(logits)
    # print(softmax)
    N, C = list(softmax.size())
    one_hot_label = F.one_hot(label, C)#.to(logits.device)
    # print(one_hot_label)
    label_prob = torch.masked_select(softmax, one_hot_label.bool())
    max_non_label_info = torch.max(softmax-one_hot_label, dim=1)
    max_non_label_prob, max_non_label_cls = max_non_label_info.values, max_non_label_info.indices
    loss_margin = margin
    # print(softmax)
    # print(label)
    # print(label_prob)
    # print(max_non_label_prob)

    l_1 = torch.tensor([0.0]*N).to(softmax.device)
    lm = max_non_label_prob - label_prob  + loss_margin
    l_2 = lm ** 2 / loss_margin
    l_3 = lm
    adv_loss = torch.mean(torch.max(l_1, torch.min(l_2, l_3)))
    return loss_cls, label_prob, max_non_label_prob, max_non_label_cls

def targeted_pert_loss(logits, label, margin=0.05):
    """Args:
        logits (torch.Tensor): logits.
        label (torch.Tensor): The target label.
        kwargs: Any keyword argument to be used to calculate
            CrossEntropy loss.

    Returns:
        torch.Tensor: The returned CrossEntropy loss.
    """
    # cls_score #[N,C]
    # label N
    logits = logits.to(label.device)
    loss_cls = F.cross_entropy(logits, label)
    softmax = F.softmax(logits)
    # print(softmax)
    N, C = list(softmax.size())
    one_hot_label = F.one_hot(label, C)#.to(logits.device)
    # print(one_hot_label)
    label_prob = torch.masked_select(softmax, one_hot_label.bool())
    max_non_label_info = torch.max(softmax-one_hot_label, dim=1)
    max_non_label_prob, max_non_label_cls = max_non_label_info.values, max_non_label_info.indices
    loss_margin = margin
    # print(softmax)
    # print(label)
    # print(label_prob)
    # print(max_non_label_prob)

    l_1 = torch.tensor([0.0]*N).to(softmax.device)
    lm = max_non_label_prob - label_prob  + loss_margin
    l_2 = lm ** 2 / loss_margin
    l_3 = lm
    adv_loss = torch.mean(torch.max(l_1, torch.min(l_2, l_3)))
    return loss_cls, label_prob, max_non_label_prob, max_non_label_cls

def untargeted_cw2_pert_loss(logits, label, kappa=0):
    N, C = list(logits.size())
    probs = F.softmax(logits, dim=1)
    label = label.item()
    onehot = torch.zeros(1, C).to(logits.device)
    onehot[0, label] = 1
    real = (onehot * probs).sum(1)[0]

    sort_prob, sort_class = probs.sort()
    second_prob = sort_prob[0][-2].unsqueeze(0)
    second_class = sort_class[0][-2].unsqueeze(0)
    return torch.clamp(torch.sum(probs) - second_prob,
                       kappa), real,  second_prob, second_class


def get_g_star(query_img, label, recognizer_model, loss_func, config_rec):
    recognizer_model.zero_grad()
    if 'tsm' in config_rec:
        query = torch.transpose(query_img, 1, 2)
    else:
        query = torch.unsqueeze(query_img, 1)
    #print('get_g_star')
    query = torch.tensor(query, requires_grad=True)
    logits = recognizer_model(query, label=None, return_loss=False, return_logit=True)
    loss, _, _, _ = loss_func(logits, label)
    loss.backward()
    return query.grad



def perturbation_image(recognizer_model, feeddict, label2cls, config_rec, logger,
                       g=None, m=None, targeted=False,
                       max_query=60000,
                       hh=0.025, # flow_lr
                       epsilon=0.1,  # fd_eta
                       delta=0.1,  # exploration
                       eta=0.1,  # online_lr
                       max_p=10,
                       target_label=None,
                       return_loss=False,
                       return_g=False,
                       return_adv=False,
                       return_inner_production=False,
                       return_inner_production_first=False,
                       loss_name='flicker_loss'):

    orig_video = feeddict['imgs1'] #[1,3,16,112,112]
    adv_video = orig_video.clone()
    label = feeddict['label1']
    _, c, t, h, w = list(orig_video.size())
    device = orig_video.device
    num_query = 0
    loop = 0
    orig_label = label.item()

    if targeted:
        max_query = 200000
        if target_label is None:
            target_label = random.sample([k for k in range(len(label2cls)) if k != orig_label], 1)[0]
        label = torch.tensor([target_label]).to(device)
        logger.info('\t  {:s}  --> {:s} '.format(label2cls[orig_label], label2cls[target_label]))

    if targeted:
        loss_func = targeted_pert_loss
    else:
        if loss_name == 'flicker_loss':
            loss_func = untargeted_pert_loss
        elif loss_name == 'cw2_loss':
            loss_func = untargeted_cw2_pert_loss
        elif loss_name == 'cross_entropy_loss':
            loss_func = untargeted_cross_entropy_pert_loss
        else:
            print('{:s} not a valid loss name')
            raise

    if g is None:
        g = torch.zeros([1, c, t, h, w]).to(device)
    if m is None:
        mv = feeddict['flow1'] #[1, 2, 16, 112, 112]
    loss_list = []
    est_g_deriv_list = []
    inner_production_list = []
    while num_query < max_query:
        # rs_frame = torch.randn(t, c, h, w).to(device)
        rs_frame = torch.randn(1, c, h, w).to(device)
        if m is None:
            rs_frames = wrap_with_optical_flow(rs_frame, mv) # [1, c, t, h, w]
        else:
            print(' Not implemented yet')
            raise

        delta_rs_frame = delta * rs_frames
        w1 = g + delta_rs_frame #[1, c, t, h, w]
        w2 = g - delta_rs_frame
        # print(delta_rs_frame)
        query_img = []
        #print(torch.max(255 * epsilon*w1/norm2(w1)), torch.min(255 * epsilon*w1/norm2(w1)))
        query_img.append(adv_video + 255 * epsilon*w1/norm2(w1))
        query_img.append(adv_video + 255 * epsilon*w2/norm2(w2))
        query_img.append(adv_video)
        adv_video_test = torch.clone(adv_video)
        query_logits=[]
        with torch.no_grad():
            for i in range(3):
                if 'tsm' in config_rec:
                    query = torch.transpose(query_img[i], 1, 2)
                else:
                    query = torch.unsqueeze(query_img[i], 1)
                logits = recognizer_model(query, label=None, return_loss=False)
                query_logits.append(torch.tensor(logits))
        l1, _, _, _, = loss_func(query_logits[0], label) # untargeted_pert_loss(query_logits[0], label)
        l2, _, _, _, = loss_func(query_logits[1], label) # untargeted_pert_loss(query_logits[1], label)
        #print(l1, l2)
        loss, label_prob, max_non_label_prob, max_non_label_cls = loss_func(query_logits[2], label) # # untargeted_pert_loss(query_logits[2], label)
        loss_list.append(loss.item())
        num_query += 3
        est_g_deriv = (l1-l2)/(delta*epsilon) # (delta*epsilon*epsilon)
        est_g_deriv_list.append(est_g_deriv.item())
        est_g_deriv = est_g_deriv.item() * rs_frames # -?
        g += eta*est_g_deriv
        if return_inner_production_first:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)
            unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            inner_production_value = torch.dot(unit_g_start, unit_g).item()
            sign_inner_production_value = torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item()
            return inner_production_value, sign_inner_production_value

        if return_inner_production:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)

            # unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            # unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            # inner_production_list.append(torch.dot(unit_g_start, unit_g).item())
            inner_production_list.append(torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item())


        adv_video = adv_video - 255 * hh * g.sign()
        adv_video = torch.max(torch.min(adv_video, orig_video+max_p), orig_video-max_p)
        pred_adv_label = query_logits[2].argmax()
        if (loop % 1000 == 0) or (not targeted and pred_adv_label != orig_label) or (targeted and pred_adv_label == target_label):
            if not targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(), label2cls[orig_label], label_prob.item(),  # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item()))
            if targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(),  # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item(),
                        label2cls[target_label], label_prob.item()))
        loop += 1
        if not targeted and pred_adv_label != orig_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_adv:
                return_list += [adv_video_test]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
        if targeted and pred_adv_label == target_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_adv:
                return_list += [adv_video_test]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list

    return_list = [pred_adv_label, num_query, False]
    if return_loss:
        return_list += [loss_list]
    if return_g:
        return_list += [est_g_deriv_list]
    if return_adv:
        return_list += [adv_video_test]
    if return_inner_production:
        return_list += [inner_production_list]
    return return_list


def perturbation_image_decompose(recognizer_model, feeddict, label2cls, config_rec, logger,
                                 code_p_func, transform_type,
                                 update_list=[1,1],
                       p_static=None, p_motion=None,
                       max_query=60000,
                       hh=0.025, # flow_lr
                       epsilon=0.1,  # fd_eta
                       delta=0.1,  # exploration
                       eta=0.1,  # online_lr
                       max_p=10):

    orig_video = feeddict['imgs1'] #[1,3,16,112,112]
    adv_video = orig_video.clone()
    label = feeddict['label1']
    _, c, t, h, w = list(orig_video.size())
    device = orig_video.device
    num_query = 0
    loop = 0

    assert p_static is not None  # [1, 3, 1, H = 112, W = 112]
    assert p_motion is not None  # [1, 16,3,3]
    # p_static = p_static.to(device)
    # p_motion = p_motion.to(device)
    g = code_p_func(p_static, p_motion) * 10. / 255.
    p_static_loop_max, p_motion_loop_max = update_list
    p_motion_transform = PerspectiveRegressionHead(transform_type=transform_type, in_channels=256)

    while num_query < max_query:
        # perturb p_static
        rs_frame = torch.randn(1, c, 1, h, w).to(device)
        rs_motion = torch.randn(1, t, p_motion_transform.num_freedom).to(device)
        rs_motion = p_motion_transform.vector2perspectiveM(rs_motion)
        rs_frames = code_p_func(rs_frame, rs_motion)
        delta_rs_frame = delta * rs_frames
        w1 = g + delta_rs_frame  # [1, c, t, h, w]
        w2 = g - delta_rs_frame
        query_img = []
        query_img.append(adv_video + 255 * epsilon * w1 / norm2(w1))
        query_img.append(adv_video + 255 * epsilon * w2 / norm2(w2))
        query_img.append(adv_video)
        query_logits = []
        with torch.no_grad():
            for i in range(3):
                if 'tsm' in config_rec:
                    query = torch.transpose(query_img[i], 1, 2)
                else:
                    query = torch.unsqueeze(query_img[i], 1)
                logits = recognizer_model(query, label=None, return_loss=False)
                query_logits.append(torch.tensor(logits))
        l1, _, _, _, = untargeted_pert_loss(query_logits[0], label)  # untargeted_pert_loss(query_logits[0], label)
        l2, _, _, _, = untargeted_pert_loss(query_logits[1], label)  # untargeted_pert_loss(query_logits[1], label)
        # print(l1, l2)
        loss, label_prob, max_non_label_prob, max_non_label_cls = untargeted_pert_loss(query_logits[2],
                                                                                       label)  # # untargeted_pert_loss(query_logits[2], label)
        num_query += 3
        est_g_deriv = (l1 - l2) / (delta * epsilon)  # (delta*epsilon*epsilon)
        est_g_deriv = est_g_deriv.item() * rs_frames  # -?
        g += eta * est_g_deriv

        adv_video = adv_video - 255 * hh * g.sign()
        adv_video = torch.max(torch.min(adv_video, orig_video + max_p), orig_video - max_p)
        pred_adv_label = query_logits[2].argmax()
        orig_label = label.item()
        if (loop % 1000 == 0) or pred_adv_label != orig_label:
            logger.info('\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                num_query, max_query, loss.item(), label2cls[orig_label], label_prob.item(),  # ))#
                label2cls[max_non_label_cls.item()], max_non_label_prob.item()))
        loop += 1
        if pred_adv_label != orig_label:
            return pred_adv_label, num_query - 2, True
    return pred_adv_label, num_query, False


def perturbation_image_decompose_from_scratch(recognizer_model, feeddict, label2cls, config_rec, logger,
                                               code_p_func, transform_type,
                                               g = None, m=None, targeted=False,
                                               max_query=60000,
                                               hh=0.025, # flow_lr
                                               epsilon=0.1,  # fd_eta
                                               delta=0.1,  # exploration
                                               eta=0.1,  # online_lr
                                               max_p=10,
                                               target_label=None,
                                               return_loss=False,
                                               return_g=False,
                                               return_adv=False,
                                               return_inner_production=False,
                                               return_inner_production_first=False,
                                               loss_name = 'flicker_loss'):


    orig_video = feeddict['imgs1'] #[1,3,16,112,112]
    adv_video = orig_video.clone()
    label = feeddict['label1']
    _, c, t, h, w = list(orig_video.size())
    device = orig_video.device
    num_query = 0
    loop = 0
    orig_label = label.item()
    if targeted:
        max_query = 200000
        if target_label is None:
            target_label = random.sample([k for k in range(len(label2cls)) if k != orig_label], 1)[0]
        label = torch.tensor([target_label]).to(device)
        logger.info('\t  {:s}  --> {:s} '.format(label2cls[orig_label], label2cls[target_label]))
    if targeted:
        loss_func = targeted_pert_loss
    else:
        if loss_name == 'flicker_loss':
            loss_func = untargeted_pert_loss
        elif loss_name == 'cw2_loss':
            loss_func = untargeted_cw2_pert_loss
        elif loss_name == 'cross_entropy_loss':
            loss_func = untargeted_cross_entropy_pert_loss
        else:
            print('{:s} not a valid loss name')
            raise

    if g is None:
        g = torch.zeros([1, c, t, h, w]).to(device)
    assert m is None
    p_motion_transform = PerspectiveRegressionHead(transform_type=transform_type, in_channels=256)

    loss_list = []
    est_g_deriv_list = []
    inner_production_list = []
    while num_query < max_query:
        rs_frame = torch.randn(1, c, 1, h, w).to(device)
        rs_motion = torch.randn(1, t, p_motion_transform.num_freedom).to(device)
        rs_motion = p_motion_transform.vector2perspectiveM(rs_motion)
        rs_frames = code_p_func(rs_frame, rs_motion)
        delta_rs_frame = delta * rs_frames
        w1 = g + delta_rs_frame  # [1, c, t, h, w]
        w2 = g - delta_rs_frame
        query_img = []
        query_img.append(adv_video + 255 * epsilon * w1 / norm2(w1))
        query_img.append(adv_video + 255 * epsilon * w2 / norm2(w2))
        query_img.append(adv_video)
        adv_video_test = torch.clone(adv_video)
        query_logits = []
        with torch.no_grad():
            for i in range(3):
                if 'tsm' in config_rec:
                    query = torch.transpose(query_img[i], 1, 2)
                else:
                    query = torch.unsqueeze(query_img[i], 1)
                logits = recognizer_model(query, label=None, return_loss=False)
                query_logits.append(torch.tensor(logits))
        l1, _, _, _, = loss_func(query_logits[0], label)  # untargeted_pert_loss(query_logits[0], label)
        l2, _, _, _, = loss_func(query_logits[1], label)  # untargeted_pert_loss(query_logits[1], label)
        # print(l1, l2)
        loss, label_prob, max_non_label_prob, max_non_label_cls = loss_func(query_logits[2],
                                                                                       label)  # # untargeted_pert_loss(query_logits[2], label)
        loss_list.append(loss.item())
        num_query += 3
        est_g_deriv = (l1 - l2) / (delta * epsilon)  # (delta*epsilon*epsilon)
        est_g_deriv_list.append(est_g_deriv.item())
        est_g_deriv = est_g_deriv.item() * rs_frames  # -?
        g += eta * est_g_deriv
        if return_inner_production_first:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)
            unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            inner_production_value = torch.dot(unit_g_start, unit_g).item()
            sign_inner_production_value = torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item()
            return inner_production_value, sign_inner_production_value

        if return_inner_production:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)

            # unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            # unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            # inner_production_list.append(torch.dot(unit_g_start, unit_g).item())
            inner_production_list.append(
                torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item())
        adv_video = adv_video - 255 * hh * g.sign()
        adv_video = torch.max(torch.min(adv_video, orig_video + max_p), orig_video - max_p)
        pred_adv_label = query_logits[2].argmax()
        if (loop % 1000 == 0) or (not targeted and pred_adv_label != orig_label) or (targeted and pred_adv_label == target_label):
            if not targeted:
                logger.info('\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                    num_query, max_query, loss.item(), label2cls[orig_label], label_prob.item(),  # ))#
                    label2cls[max_non_label_cls.item()], max_non_label_prob.item()))
            if targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(),   # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item(),
                        label2cls[target_label], label_prob.item()))
        loop += 1
        if not targeted and  pred_adv_label != orig_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_adv:
                return_list += [adv_video_test]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
        if targeted and  pred_adv_label == target_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_adv:
                return_list += [adv_video_test]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
    return_list = [pred_adv_label, num_query, False]
    if return_loss:
        return_list += [loss_list]
    if return_g:
        return_list += [est_g_deriv_list]
    if return_adv:
        return_list += [adv_video_test]
    if return_inner_production:
        return_list += [inner_production_list]
    return return_list


def perturbation_image_multi_noise(recognizer_model, feeddict, label2cls, config_rec, logger,
                       g=None, m=None, targeted=False,
                       max_query=60000,
                       hh=0.025, # flow_lr
                       epsilon=0.1,  # fd_eta
                       delta=0.1,  # exploration
                       eta=0.1,  # online_lr
                       max_p=10,
                       target_label=None,
                       return_loss=False,
                       return_g=False,
                       return_inner_production=False,
                       return_inner_production_first=False,
                       loss_name='flicker_loss'):

    orig_video = feeddict['imgs1'] #[1,3,16,112,112]
    adv_video = orig_video.clone()
    label = feeddict['label1']
    _, c, t, h, w = list(orig_video.size())
    device = orig_video.device
    num_query = 0
    loop = 0
    orig_label = label.item()
    if targeted:
        max_query = 200000
        if target_label is None:
            target_label = random.sample([k for k in range(len(label2cls)) if k != orig_label], 1)[0]
        label = torch.tensor([target_label]).to(device)
        logger.info('\t  {:s}  --> {:s} '.format(label2cls[orig_label], label2cls[target_label]))

    if targeted:
        loss_func = targeted_pert_loss
    else:
        if loss_name == 'flicker_loss':
            loss_func = untargeted_pert_loss
        elif loss_name == 'cw2_loss':
            loss_func = untargeted_cw2_pert_loss
        elif loss_name == 'cross_entropy_loss':
            loss_func = untargeted_cross_entropy_pert_loss
        else:
            print('{:s} not a valid loss name')
            raise

    if g is None:
        g = torch.zeros([1, c, t, h, w]).to(device)
    loss_list = []
    est_g_deriv_list = []
    inner_production_list = []
    while num_query < max_query:
        rs_frames = torch.randn(1, c, t,  h, w).to(device)
        delta_rs_frame = delta * rs_frames
        w1 = g + delta_rs_frame #[1, c, t, h, w]
        w2 = g - delta_rs_frame
        # print(delta_rs_frame)
        query_img = []
        #print(torch.max(255 * epsilon*w1/norm2(w1)), torch.min(255 * epsilon*w1/norm2(w1)))
        query_img.append(adv_video + 255 * epsilon*w1/norm2(w1))
        query_img.append(adv_video + 255 * epsilon*w2/norm2(w2))
        query_img.append(adv_video)
        query_logits=[]
        with torch.no_grad():
            for i in range(3):
                if 'tsm' in config_rec:
                    query = torch.transpose(query_img[i], 1, 2)
                else:
                    query = torch.unsqueeze(query_img[i], 1)
                logits = recognizer_model(query, label=None, return_loss=False)
                query_logits.append(torch.tensor(logits))
        l1, _, _, _, = loss_func(query_logits[0], label) # untargeted_pert_loss(query_logits[0], label)
        l2, _, _, _, = loss_func(query_logits[1], label) # untargeted_pert_loss(query_logits[1], label)
        #print(l1, l2)
        loss, label_prob, max_non_label_prob, max_non_label_cls = loss_func(query_logits[2], label) # # untargeted_pert_loss(query_logits[2], label)
        loss_list.append(loss.item())
        num_query += 3
        est_g_deriv = (l1-l2)/(delta*epsilon) # (delta*epsilon*epsilon)
        est_g_deriv_list.append(est_g_deriv.item())
        est_g_deriv = est_g_deriv.item() * rs_frames # -?
        g += eta*est_g_deriv
        if return_inner_production_first:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)
            unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            inner_production_value = torch.dot(unit_g_start, unit_g).item()
            sign_inner_production_value = torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item()
            return inner_production_value, sign_inner_production_value
        if return_inner_production:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)

            # unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            # unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            # inner_production_list.append(torch.dot(unit_g_start, unit_g).item())
            inner_production_list.append(
                torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item())
        adv_video = adv_video - 255 * hh * g.sign()
        adv_video = torch.max(torch.min(adv_video, orig_video+max_p), orig_video-max_p)
        pred_adv_label = query_logits[2].argmax()
        if (loop % 1000 == 0) or (not targeted and pred_adv_label != orig_label) or (targeted and pred_adv_label == target_label):
            if not targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(), label2cls[orig_label], label_prob.item(),  # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item()))
            if targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(),  # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item(),
                        label2cls[target_label], label_prob.item()))
        loop += 1
        if not targeted and pred_adv_label != orig_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
        if targeted and pred_adv_label == target_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
    return_list = [pred_adv_label, num_query, False]
    if return_loss:
        return_list += [loss_list]
    if return_g:
        return_list += [est_g_deriv_list]
    if return_inner_production:
        return_list += [inner_production_list]
    return return_list

def perturbation_image_one_noise(recognizer_model, feeddict, label2cls, config_rec, logger,
                       g=None, m=None, targeted=False,
                       max_query=60000,
                       hh=0.025, # flow_lr
                       epsilon=0.1,  # fd_eta
                       delta=0.1,  # exploration
                       eta=0.1,  # online_lr
                       max_p=10,
                       target_label=None,
                       return_loss=False,
                       return_g=False,
                       return_inner_production=False,
                       return_inner_production_first=False,
                       loss_name='flicker_loss'):

    orig_video = feeddict['imgs1'] #[1,3,16,112,112]
    adv_video = orig_video.clone()
    label = feeddict['label1']
    _, c, t, h, w = list(orig_video.size())
    device = orig_video.device
    num_query = 0
    loop = 0
    orig_label = label.item()
    if targeted:
        max_query = 200000
        if target_label is None:
            target_label = random.sample([k for k in range(len(label2cls)) if k != orig_label], 1)[0]
        label = torch.tensor([target_label]).to(device)
        logger.info('\t  {:s}  --> {:s} '.format(label2cls[orig_label], label2cls[target_label]))

    if targeted:
        loss_func = targeted_pert_loss
    else:
        if loss_name == 'flicker_loss':
            loss_func = untargeted_pert_loss
        elif loss_name == 'cw2_loss':
            loss_func = untargeted_cw2_pert_loss
        elif loss_name == 'cross_entropy_loss':
            loss_func = untargeted_cross_entropy_pert_loss
        else:
            print('{:s} not a valid loss name')
            raise


    if g is None:
        g = torch.zeros([1, c, t, h, w]).to(device)

    loss_list = []
    est_g_deriv_list = []
    inner_production_list = []
    while num_query < max_query:
        rs_frame = torch.randn(1, c, h, w).to(device)
        rs_frames = torch.unsqueeze(rs_frame, dim=-3).repeat([1, 1, t, 1, 1])
        delta_rs_frame = delta * rs_frames
        w1 = g + delta_rs_frame #[1, c, t, h, w]
        w2 = g - delta_rs_frame
        # print(delta_rs_frame)
        query_img = []
        #print(torch.max(255 * epsilon*w1/norm2(w1)), torch.min(255 * epsilon*w1/norm2(w1)))
        query_img.append(adv_video + 255 * epsilon*w1/norm2(w1))
        query_img.append(adv_video + 255 * epsilon*w2/norm2(w2))
        query_img.append(adv_video)
        query_logits=[]
        with torch.no_grad():
            for i in range(3):
                if 'tsm' in config_rec:
                    query = torch.transpose(query_img[i], 1, 2)
                else:
                    query = torch.unsqueeze(query_img[i], 1)
                logits = recognizer_model(query, label=None, return_loss=False)
                query_logits.append(torch.tensor(logits))
        l1, _, _, _, = loss_func(query_logits[0], label) # untargeted_pert_loss(query_logits[0], label)
        l2, _, _, _, = loss_func(query_logits[1], label) # untargeted_pert_loss(query_logits[1], label)
        #print(l1, l2)
        loss, label_prob, max_non_label_prob, max_non_label_cls = loss_func(query_logits[2], label) # # untargeted_pert_loss(query_logits[2], label)
        loss_list.append(loss.item())
        num_query += 3
        est_g_deriv = (l1-l2)/(delta*epsilon) # (delta*epsilon*epsilon)
        est_g_deriv_list.append(est_g_deriv.item())
        est_g_deriv = est_g_deriv.item() * rs_frames # -?
        g += eta*est_g_deriv
        if return_inner_production_first:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)
            unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            inner_production_value = torch.dot(unit_g_start, unit_g).item()
            sign_inner_production_value = torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item()
            return inner_production_value, sign_inner_production_value
        if return_inner_production:
            g_star = get_g_star(query_img=adv_video,
                                label=label,
                                recognizer_model=recognizer_model,
                                loss_func=loss_func,
                                config_rec=config_rec)

            # unit_g_start = F.normalize(g_star.contiguous().view(g_star.size(0), -1), p=2, dim=1).view(-1)
            # unit_g = F.normalize(g.view(g.size(0), -1), p=2, dim=1).view(-1)
            # inner_production_list.append(torch.dot(unit_g_start, unit_g).item())
            inner_production_list.append(
                torch.dot(g_star.sign().contiguous().view(-1), g.sign().contiguous().view(-1)).item())
            #g = g_star.squeeze(dim=0)
        adv_video = adv_video - 255 * hh * g.sign()
        adv_video = torch.max(torch.min(adv_video, orig_video+max_p), orig_video-max_p)
        pred_adv_label = query_logits[2].argmax()
        if (loop % 1000 == 0) or (not targeted and pred_adv_label != orig_label) or (targeted and pred_adv_label == target_label):
            if not targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(), label2cls[orig_label], label_prob.item(),  # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item()))
            if targeted:
                logger.info(
                    '\t inner loop [{:d}/{:d}]: loss {:1.3f}, {:s} {:1.2f}  --> {:s} {:1.2f}'.format(  # '.format( #
                        num_query, max_query, loss.item(),  # ))#
                        label2cls[max_non_label_cls.item()], max_non_label_prob.item(),
                        label2cls[target_label], label_prob.item()))
        loop += 1
        if not targeted and pred_adv_label != orig_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
        if targeted and pred_adv_label == target_label:
            return_list = [pred_adv_label, num_query - 2, True]
            if return_loss:
                return_list += [loss_list]
            if return_g:
                return_list += [est_g_deriv_list]
            if return_inner_production:
                return_list += [inner_production_list]
            return return_list
    return_list = [pred_adv_label, num_query, False]
    if return_loss:
        return_list += [loss_list]
    if return_g:
        return_list += [est_g_deriv_list]
    if return_inner_production:
        return_list += [inner_production_list]
    return return_list



if __name__ == "__main__":
    x = torch.tensor([[11,12,13],
                      [21,22,23]]).float() #[2,3]
    x = x.unsqueeze(dim=0).unsqueeze(dim=0) # [1, 1, 2, 3]
    optical_flow_x = torch.tensor([[0.2,0.2,0.2],
                                   [0.2,0.2,0.2]])
    optical_flow_y = torch.tensor([[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]])
    optical_flow_x = torch.tensor([[0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5]])

    optical_flow = torch.stack([optical_flow_x, optical_flow_y],dim=0) #[2,2,3]
    optical_flow = optical_flow.unsqueeze(0).unsqueeze(dim=-3) #[1,2,1,2,3]
    print(optical_flow.size())
    out_frame = wrap_with_optical_flow(x, optical_flow) #[1,1,1,2,3]
    print(out_frame.squeeze())




















