#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
import os, time
from random import randint
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import ImageFilter
import torchvision.transforms as transforms
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from, websockets, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    all_edges = []
    for view in scene.getTrainCameras():
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())
    my_viewpoint_stack = scene.getTrainCameras().copy()
    edges_stack = all_edges.copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render(cam, gaussians, pipe, background, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "N_GS": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration == 300:
                gaussians.only_prune(0.02)

            # Densification
            if opt.densify_from_iter < iteration < opt.densify_until_iter:
                gaussians.add_densification_stats_abs(viewspace_point_tensor, visibility_filter)

                if iteration % opt.densification_interval == 0:

                    num_cams = args.cams
                    if args.cams == -1 or (iteration % 3000 == 400 and iteration < 9000):
                        num_cams = len(scene.getTrainCameras().copy())
                    edge_losses = []
                    camlist = []
                    for _ in range(num_cams):
                        if not my_viewpoint_stack:
                            my_viewpoint_stack = scene.getTrainCameras().copy()
                            edges_stack = all_edges.copy()
                        camlist.append(my_viewpoint_stack.pop())
                        edge_losses.append(edges_stack.pop())
                    gaussian_importance = compute_edge_score(camlist, edge_losses, gaussians, pipe, bg)

                    startI = opt.densify_from_iter
                    endI = opt.densify_until_iter - 500
                    rate = (iteration - startI) / (endI - startI)
                    if rate >= 1:
                        budget = int(opt.budget)
                    else:
                        budget = int(math.sqrt(rate) * opt.budget)

                    gaussians.densify_and_prune_Improved(gaussian_importance, 0.005, budget, opt, iteration)
                
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity(0.05)

                if iteration % opt.opacity_reset_interval == 300 and iteration < 9000:
                    gaussians.only_prune(0.2, True)

            # Optimizer step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    if iteration <= 15000:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        gaussians.shoptimizer.step()
                        gaussians.shoptimizer.zero_grad(set_to_none=True)
                    elif iteration <= 22500:
                        if iteration % 5 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                    else:
                        if iteration % 20 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)


def get_edges(image):
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    return image_edges_tensor


def normalize(value_tensor):
    value_tensor[value_tensor.isnan()] = 0
    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = valid_value / torch.mean(valid_value)

    return ret_value


def compute_edge_score(camlist, edge_losses, gaussians, pipe, bg):
    num_points = len(gaussians.get_xyz)
    gaussian_importance = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    visibility_filter_all = torch.zeros(num_points, device="cuda", dtype=bool)
    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        pixel_weights = edge_losses[view].cuda()
        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights=pixel_weights)

        loss_accum = normalize(render_pkg["accum_weights"])
        visibility_filter = render_pkg["visibility_filter"].detach()

        gaussian_importance[visibility_filter] += loss_accum[visibility_filter] / len(camlist)
        visibility_filter_all[visibility_filter] = True

    gaussian_importance[visibility_filter_all] = gaussian_importance[visibility_filter_all]
    return gaussian_importance


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test= 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000 * (i + 1) for i in range(10)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cams", type=int, default=10)
    parser.add_argument("--websockets", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations,
        args.debug_from, 
        args.websockets,
        args
    )

    # All done
    print("\nTraining complete.")
