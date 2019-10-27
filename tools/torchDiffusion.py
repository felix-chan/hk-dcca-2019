"""
Diffusion object using PyTorch to speech up the calculation
"""

import torch
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LinearRing, MultiPolygon

class torchDiffusion:

    boundary = None
    init_pts = None
    last_pts = None
    point_map = None
    pt_force = 1
    boundary_force = 1
    dt = 1
    vx = None
    vy = None
    max_force = 5

    step_history = []

    def __init__(self, points, boundary, pt_force, boundary_force, 
        max_force = 5, dt = 1):
        # Check input point DataFrame column
        assert('name' in points.columns)
        assert('x' in points.columns)
        assert('y' in points.columns)

        # Set up mass of object
        if not 'mass' in points.columns:
            points['mass'] = 1

        self.init_pts = points
        self.last_pts = points
        self.boundary = boundary
        self.pt_force = pt_force
        self.boundary_force = boundary_force
        self.max_force = max_force
        self.dt = dt

        total_groups = len(points.name.unique())
        self.point_map = pd.DataFrame({
            'name': points.name.unique(),
            'id': range(total_groups)
        })

        self.vx, self.vy = torch.zeros(total_groups).double(), \
            torch.zeros(total_groups).double()
        self.step_history = []

    def _cal_new_force(self, x1, x2, y1, y2, group, f_mat=1):
        distance = (x2 - x1)**2 + (y2 - y1)**2
        force = (np.exp(1/(distance / f_mat + 0.5)) - 1)
        net_x = (force / distance * (x1 - x2)).double()
        net_y = (force / distance * (y1 - y2)).double()
        
        # Group force
        M = torch.zeros(int(group.max()+1), len(net_x))
        M[group.long(), torch.arange(len(net_x))] = 1
        net_x_sum = torch.mm(M.double(), net_x.view(len(net_x), 1)).view(-1)
        net_y_sum = torch.mm(M.double(), net_y.view(len(net_x), 1)).view(-1)

        net_x_sum = net_x_sum * self.max_force
        net_y_sum = net_y_sum * self.max_force
        
        return net_x_sum, net_y_sum

    def _single_boundary_force(self, x, y, boundary):
        r = self.boundary_force
        pts = Point(x, y)
        boundary_ext = LinearRing(boundary.exterior.coords)
        d = boundary_ext.project(pts)
        p = boundary_ext.interpolate(d)
        px, py = list(p.coords)[0]
        
        f = abs(min(np.log(d/r + 0.000001), 0)) * len(self.point_map['id'])
        ax, ay = (x - px) * f / d, (y - py) * f / d
        return ax, ay

    def _boundary_force(self, x1, y1):
        bd_f = [self._single_boundary_force(x, y, self.boundary) for 
                x, y in zip(x1.numpy(), y1.numpy())]

        return torch.tensor([x[0] for x in bd_f], dtype=torch.float64), \
            torch.tensor([x[1] for x in bd_f], dtype=torch.float64)

    def _check_inside(self, x_list, y_list):
        return [Point(x,y).within(self.boundary) for x,y in zip(x_list, y_list)]

    def forward(self, feedback = False):
        # Calculate the interpoint force 
        point_list = self.last_pts[['name', 'x', 'y']].copy()
        point_list['dummy_key'] = 1
        point_comb = point_list.rename({
            'x': 'x1',
            'y': 'y1'
        }, axis=1).merge(
            point_list.rename({
                'name': 'name2',
                'x': 'x2',
                'y': 'y2'
            }, axis=1),
            on='dummy_key'
        )
        point_comb = point_comb[point_comb.name != point_comb.name2].merge(
            self.point_map,
            on='name'
        )

        # Force add by other points
        point_fx, point_fy = self._cal_new_force(
            torch.from_numpy(point_comb['x1'].values),
            torch.from_numpy(point_comb['x2'].values),
            torch.from_numpy(point_comb['y1'].values),
            torch.from_numpy(point_comb['y1'].values),
            torch.from_numpy(point_comb['id'].values),
            self.pt_force
        )

        # Force add by boundary
        point_v_list = self.last_pts[['name', 'x', 'y', 'mass']].merge(
            self.point_map,
            on='name'
        ).sort_values(['id'])
        boundary_fx, boundary_fy = self._boundary_force(
            torch.from_numpy(point_list['x'].values),
            torch.from_numpy(point_list['y'].values)
        )

        # New point movement veolcity
        point_mass = torch.from_numpy(point_v_list['mass'].values).double()
        tfdt = torch.ones((len(point_v_list['mass'].values)), 
            dtype=torch.double) * float(self.dt)
        new_vx = self.vx * 0.3 + (point_fx + boundary_fx).double() / \
            point_mass * tfdt.double()
        new_vy = self.vy * 0.3 + (point_fy + boundary_fy).double() / \
            point_mass * tfdt.double()

        # Calculate new point location
        new_dx = torch.from_numpy(point_list['x'].values).double() + \
            new_vx * tfdt.double()
        new_dy = torch.from_numpy(point_list['y'].values).double() + \
            new_vy * tfdt.double()

        check_inside = self._check_inside(new_dx.numpy(), new_dy.numpy())

        self.vx, self.vy = new_vx * torch.Tensor(check_inside).double(), \
            new_vy * torch.Tensor(check_inside).double()
        point_v_list['x'] = np.where(check_inside, new_dx.numpy(), 
                                     point_v_list['x'])
        point_v_list['y'] = np.where(check_inside, new_dy.numpy(), 
                                     point_v_list['y'])
        point_v_list['can_move'] = check_inside

        self.last_pts = point_v_list.copy()
        self.step_history.append(point_v_list.copy())

        if feedback:
            return point_v_list
        else:
            return None

    def get_steps(self):
        return self.step_history