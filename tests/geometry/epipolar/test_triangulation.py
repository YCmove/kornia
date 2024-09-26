from typing import Dict

import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.geometry.epipolar as epi
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from testing.base import assert_close
# from testing.base import BaseTester

class TestTriangulation:
    # def test_smoke(self, device, dtype):
    #     P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
    #     P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
    #     points1 = torch.rand(1, 1, 2, device=device, dtype=dtype)
    #     points2 = torch.rand(1, 1, 2, device=device, dtype=dtype)
    #     points3d = epi.triangulate_points(P1, P2, points1, points2)

    #     # print(f'P1={P1}')
    #     # print(f'P2={P2}')
    #     # print(f'points1={points1}')
    #     # print(f'points2={points2}')
    #     # print(f'points3d={points3d}')
    #     assert points3d.shape == (1, 1, 3)
    
    def test_find_optimal_pts_by_niter2(self, device, dtype):
        points3d = torch.tensor([
            [[0,0,1],
            [0,0.1,1],
            [0.1,0,1],
            [0.1,0.1,1]]
        ]).type(dtype).to(device)

        cam1_from_world = torch.tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0]
        ]).unsqueeze(0)
        cam2_from_world = torch.tensor([
            [1,0,0,1],
            [0,1,0,0],
            [0,0,1,0]
        ]).unsqueeze(0)

        P1 = cam1_from_world.type(dtype).to(device)
        P2 = cam2_from_world.type(dtype).to(device)

        K1, R1, t1 = kornia.geometry.KRt_from_projection(P1)
        K2, R2, t2 = kornia.geometry.KRt_from_projection(P2)
        E_mat = kornia.geometry.essential_from_Rt(R1, t1, R2, t2)

        for i in range(points3d.shape[1]):
            print()
            points3d_h = convert_points_to_homogeneous(points3d[:,i,:]).unsqueeze(2)
            points1 = convert_points_from_homogeneous((P1 @ points3d_h).transpose(1,2))
            points2 = convert_points_from_homogeneous((P2 @ points3d_h).transpose(1,2))

            points1_opt, points2_opt = epi.find_optimal_pts_by_niter2(E_mat, R1, t1, R2, t2, points1, points2)

            assert_close(points1_opt, points1)
            assert_close(points2_opt, points2)

            # print(f'------------')

    def test_data(self, device, dtype):
        # P1 = torch.tensor([
        #     [[500, 0, 320, 0],
        #     [0, 500, 240, 0],
        #     [0, 0, 1, 0]]
        # ]).type(dtype).to(device)
        # P2 = torch.tensor([
        #     [[490, 0, 330, -50],
        #     [0, 490, 250, 10],
        #     [0, 0, 1, 0]]
        # ]).type(dtype).to(device)
        # # points1 = torch.tensor([
        # #     [[250, 300], [400, 500], [350, 450]]
        # # ]).type(dtype).to(device)
        # points1 = torch.tensor([
        #     [[260, 310], [400, 500], [350, 450]]
        # ]).type(dtype).to(device)
        # points2 = torch.tensor([
        #     [[245, 290], [390, 480], [345, 440]]
        # ]).type(dtype).to(device)

        # Test data from OpenCV
        P1 = torch.tensor([
            [[250, 0, 200, 0],
            [0, 250, 150, 0],
            [0, 0, 1, 0]]
        ]).type(dtype).to(device)
        P2 = torch.tensor([
            [[246.2, 0, 43.41, 0],
            [0, 250, 150, 0],
            [-43.41, 0, 246.2, 250]]
        ]).type(dtype).to(device)
        points1 = torch.tensor([
            [[220., 130.]]
        ]).type(dtype).to(device)
        points2 = torch.tensor([
            [[190., 140.]]
        ]).type(dtype).to(device)
        # points2 = torch.tensor([
        #     [[195., 136.]]
        # ]).type(dtype).to(device)

        points1, points2, points3d = epi.triangulate_optimalpoints(P1, P2, points1, points2)

        # print(f'P1={P1}')
        # print(f'P2={P2}')
        # print(f'points1={points1}')
        # print(f'points2={points2}')
        # print(f'points3d={points3d}')

        
        # points3d = epi.triangulate_points(P1, P2, points1, points2)
        # # Batch size = 1
        # B = P1.shape[0]  # B = 1

        # # Compute the camera center for the first camera
        # # The camera center is the null space of P1 (i.e., P1 * C1 = 0)
        # C1_batch = []
        # for i in range(B):
        #     _, _, V1 = torch.linalg.svd(P1[i])
        #     C1 = V1[-1]  # Last row of V1 gives the null space
        #     C1 = C1 / C1[-1]  # Convert to inhomogeneous coordinates
        #     C1_batch.append(C1)

        # C1_batch = torch.stack(C1_batch).type(dtype).to(device)  # Shape: (B, 4)

        # # Compute the essential matrix for each batch
        # F_batch = []
        # for i in range(B):
        #     # Extract the first three columns of P2 for rotation and translation
        #     R = P2[i, :, :3]
        #     t = P2[i, :, 3]

        #     # Compute the skew-symmetric matrix of the translation vector
        #     t_skew = torch.tensor([
        #         [0, -t[2], t[1]],
        #         [t[2], 0, -t[0]],
        #         [-t[1], t[0], 0]
        #     ]).type(dtype).to(device)

        #     # Compute the essential matrix E = [t]_x * R
        #     E = (t_skew @ R).type(dtype).to(device)

        #     # Assuming intrinsic matrices are identity (for simplicity)
        #     K1 = torch.eye(3).type(dtype).to(device)  # Intrinsic matrix of the first camera
        #     K2 = torch.eye(3).type(dtype).to(device)  # Intrinsic matrix of the second camera

        #     # Compute the fundamental matrix F = K2^-T * E * K1^-1
        #     F = torch.inverse(K2).T @ E @ torch.inverse(K1)
        #     F_batch.append(F)

        # # Stack the results into a batch (shape: (B, 3, 3))
        # F_batch = torch.stack(F_batch).type(dtype).to(device)

        # print("Fundamental Matrix (F) for batch = 1:")
        # print(F_batch)

        # point_line_dist = epi.sampson_epipolar_distance(points1, points2, F_batch)
        # print(f'point_line_dist={point_line_dist}')
        # print()


    @pytest.mark.parametrize("batch_size, num_points", [(1, 3), (2, 4), (3, 5)])
    def test_shape(self, batch_size, num_points, device, dtype):
        B, N = batch_size, num_points
        P1 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        points1 = torch.rand(1, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points3d = epi.triangulate_points(P1, P2, points1, points2)
        assert points3d.shape == (B, N, 3)

    @pytest.mark.xfail()
    def test_two_view(self, device, dtype):
        num_views: int = 2
        num_points: int = 10
        scene: Dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)

        P1 = scene["P"][0:1]
        P2 = scene["P"][1:2]
        x1 = scene["points2d"][0:1]
        x2 = scene["points2d"][1:2]

        X = epi.triangulate_points(P1, P2, x1, x2)
        x_reprojected = kornia.geometry.transform_points(scene["P"], X.expand(num_views, -1, -1))

        # print(f'P1={P1}')
        # print(f'P2={P2}')
        # print(f'x1={x1}')
        # print(f'x2={x2}')
        # print(f'X={X}')
        # print(f'x_reprojected={x_reprojected}')
        # print(f'scene["points3d"]={scene["points3d"]}')
        print(f'scene-points3d={scene["points3d"].dtype}')
        print(f'X={X.dtype}')
        print(f'x_reprojected={x_reprojected.dtype}')

        assert_close(scene["points3d"], X, rtol=1e-4, atol=1e-4)
        assert_close(scene["points2d"], x_reprojected, rtol=1e-4, atol=1e-4)

    # def test_gradcheck(self, device):
    #     points1 = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
    #     points2 = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
    #     P1 = kornia.eye_like(3, points1)
    #     P1 = torch.nn.functional.pad(P1, [0, 1])
    #     P2 = kornia.eye_like(3, points2)
    #     P2 = torch.nn.functional.pad(P2, [0, 1])
    #     assert gradcheck(epi.triangulate_points, (P1, P2, points1, points2), raise_exception=True, fast_mode=True)
