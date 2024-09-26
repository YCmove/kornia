"""Module with the functionalities for triangulation."""
from typing import Tuple

import torch
import torch.nn.functional as F

import kornia
from kornia.core import zeros
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.utils.helpers import _torch_svd_cast
from .essential import find_optimal_pts_by_niter2, essential_from_Rt
from .projection import KRt_from_projection
from .fundamental import fundamental_from_essential
# from kornia.geometry.epipolar import essential_from_Rt

# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp#L68


# def find_optimal_pts_by_niter2(
#     E_mat: torch.Tensor,
#     points1: torch.Tensor,
#     points2: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     points1 = convert_points_from_homogeneous(points1)
#     points2 = convert_points_from_homogeneous(points2)
#     S = torch.Tensor([[1,0,0],[0,1,0]]).double()

#     # Epipolar lines
#     n1 = S * E_mat * points2
#     n2 = S * E_mat.T * points1

#     E_tilde = torch.zeros([2, 2]).double()
#     a = n1.T * E_tilde * n2
#     b = ((n1**2).sqrt() + (n2**2).sqrt()) / 2.0
#     c = points1.T * E_mat * points2
#     d = (b * b - a * c).sqrt()
#     lagrange = c / (b + d)

#     delta1 = lagrange * n1
#     delta2 = lagrange * n2
#     n1 -= E_tilde * delta2
#     n2 -= E_tilde.T * delta1
#     lagrange *= (2.0 * d) / (n1**2).sqrt() + (n2**2).sqrt()

#     optimal_point1 = F.normalize(points1 - S.T * lagrange * n1, dim=1)
#     optimal_point2 = F.normalize(points2 - S.T * lagrange * n2, dim=1)

#     return optimal_point1, optimal_point2


def triangulate_optimalpoints(
    P1: torch.Tensor, P2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor
) -> torch.Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.
    """
    # if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
    #     raise AssertionError(P1.shape)
    # if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
    #     raise AssertionError(P2.shape)
    # if len(P1.shape[:-2]) != len(P2.shape[:-2]):
    #     raise AssertionError(P1.shape, P2.shape)
    # if not (len(points1.shape) >= 2 and points1.shape[-1] == 2):
    #     raise AssertionError(points1.shape)
    # if not (len(points2.shape) >= 2 and points2.shape[-1] == 2):
    #     raise AssertionError(points2.shape)
    # if len(points1.shape[:-2]) != len(points2.shape[:-2]):
    #     raise AssertionError(points1.shape, points2.shape)
    # if len(P1.shape[:-2]) != len(points1.shape[:-2]):
    #     raise AssertionError(P1.shape, points1.shape)
    
    K1, R1, t1 = KRt_from_projection(P1)
    K2, R2, t2 = KRt_from_projection(P2)
    E_mat = essential_from_Rt(R1, t1, R2, t2)
    print('--------------------------')
    print(f'before points1={points1}\npoints2={points2}')
    points1, points2 = find_optimal_pts_by_niter2(E_mat, R1, t1, R2, t2, points1, points2)
    print(f'after points1={points1}\npoints2={points2}')
    print('--------------------------')

    points3d = triangulate_points(P1, P2, points1, points2)

    print(f'points3d={points3d}')

    # F_mat = fundamental_from_essential(E_mat, K1, K2)

    # point_line_dist = kornia.geometry.epipolar.sampson_epipolar_distance(points1, points2, F_mat)
    # print(f'point_line_dist={point_line_dist}')
    # print()

    return points1, points2, points3d

    # # allocate and construct the equations matrix with shape (*, 4, 4)
    # points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
    # X = zeros(points_shape[:-1] + (4, 4)).type_as(points1)

    # for i in range(4):
    #     X[..., 0, i] = points1[..., 0] * P1[..., 2:3, i] - P1[..., 0:1, i]
    #     X[..., 1, i] = points1[..., 1] * P1[..., 2:3, i] - P1[..., 1:2, i]
    #     X[..., 2, i] = points2[..., 0] * P2[..., 2:3, i] - P2[..., 0:1, i]
    #     X[..., 3, i] = points2[..., 1] * P2[..., 2:3, i] - P2[..., 1:2, i]

    # # 1. Solve the system Ax=0 with smallest eigenvalue
    # # 2. Return homogeneous coordinates

    # _, _, V = _torch_svd_cast(X)

    # points3d_h = V[..., -1]
    # points3d: torch.Tensor = convert_points_from_homogeneous(points3d_h)
    # return points3d


def triangulate_points(
    P1: torch.Tensor, P2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor
) -> torch.Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.
    """
    if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
        raise AssertionError(P1.shape)
    if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
        raise AssertionError(P2.shape)
    if len(P1.shape[:-2]) != len(P2.shape[:-2]):
        raise AssertionError(P1.shape, P2.shape)
    if not (len(points1.shape) >= 2 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if not (len(points2.shape) >= 2 and points2.shape[-1] == 2):
        raise AssertionError(points2.shape)
    if len(points1.shape[:-2]) != len(points2.shape[:-2]):
        raise AssertionError(points1.shape, points2.shape)
    if len(P1.shape[:-2]) != len(points1.shape[:-2]):
        raise AssertionError(P1.shape, points1.shape)

    # allocate and construct the equations matrix with shape (*, 4, 4)
    points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
    X = zeros(points_shape[:-1] + (4, 4)).type_as(points1)

    for i in range(4):
        X[..., 0, i] = points1[..., 0] * P1[..., 2:3, i] - P1[..., 0:1, i]
        X[..., 1, i] = points1[..., 1] * P1[..., 2:3, i] - P1[..., 1:2, i]
        X[..., 2, i] = points2[..., 0] * P2[..., 2:3, i] - P2[..., 0:1, i]
        X[..., 3, i] = points2[..., 1] * P2[..., 2:3, i] - P2[..., 1:2, i]

    # 1. Solve the system Ax=0 with smallest eigenvalue
    # 2. Return homogeneous coordinates

    _, _, V = _torch_svd_cast(X)

    points3d_h = V[..., -1]
    points3d: torch.Tensor = convert_points_from_homogeneous(points3d_h)
    return points3d
