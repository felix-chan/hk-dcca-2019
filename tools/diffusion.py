#%%
import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


#%%
# Diffusion vector
# Base class
class DiffVec(object):
    def __init__(self, target):
        self._target = target
    
    def _cal_magnitude(self, displacement):
        magnitude = 0
        return magnitude
    
    def cal_vec(self, x0, y0, x, y):
        # Displacement between two points
        dx_dy = np.array([x0 - x, y0 - y])
        displacement = np.sqrt(np.sum(dx_dy**2))
        # Normalise
        direction = dx_dy / displacement
        # Magnitude of vector
        magnitude = self._cal_magnitude(displacement)
        # Scale vector
        vx_vy = direction * magnitude
        return vx_vy


# Tanh
class DiffVecTanh(DiffVec):
    def __init__(self, target, a=1):
        DiffVec.__init__(self, target)
        self._a = a
    
    def _cal_magnitude(self, displacement):
        d = displacement - self._target
        magnitude = np.tanh(-self._a * d)
        return magnitude


# Exponential
class DiffVecExp(DiffVec):
    def __init__(self, target, l=1):
        DiffVec.__init__(self, target)
        self._l = l
    
    def _cal_magnitude(self, displacement):
        d = displacement - self._target
        magnitude = np.exp(-self._l * d)
        return magnitude


class DiffVecExp1(DiffVec):
    def __init__(self, target, l=1):
        DiffVec.__init__(self, target)
        self._l = l
    
    def _cal_magnitude(self, displacement):
        d = displacement - self._target
        magnitude = np.exp(-self._l * d) - 1
        return magnitude


class DiffVecDoubleExp(DiffVec):
    def __init__(self, target, l1=1, l2=0.5):
        DiffVec.__init__(self, target)
        self._l1 = l1
        self._l2 = l2
    
    def _cal_magnitude(self, displacement):
        d = displacement - self._target
        magnitude = np.exp(-self._l1 * d) - np.exp(-self._l2 * d)
        return magnitude


# Linear
class DiffVecLinear(DiffVec):
    def __init__(self, target, s1=0.1, s2=0.01):
        DiffVec.__init__(self, target)
        self._s1 = s1
        self._s2 = s2
    
    def _cal_magnitude(self, displacement):
        d = displacement - self._target
        magnitude = -np.where(d < 0, self._s1, self._s2) * d
        return magnitude

class DiffVecLinearCap(DiffVec):
    def __init__(self, target, s1=0.1, s2=0.01, c1=-np.inf, c2=-0.1):
        DiffVec.__init__(self, target)
        self._s1 = s1
        self._s2 = s2
        self._c1 = c1
        self._c2 = c2
    
    def _cal_magnitude(self, displacement):
        d = displacement - self._target
        magnitude = np.where(d < 0, -self._s1, -self._s2) * d
        magnitude = np.where(d < 0, np.max(magnitude, self._c1), np.min(magnitude, self._c2))
        return magnitude


# Inverse
class DiffVecInv(DiffVec):
    def __init__(self, target):
        DiffVec.__init__(self, target)
    
    def _cal_magnitude(self, displacement):
        magnitude = 1 / (displacement + self._target)
        return magnitude

class DiffVecInv1(DiffVec):
    def __init__(self, target):
        DiffVec.__init__(self, target)
    
    def _cal_magnitude(self, displacement):
        magnitude = 1 / (displacement + self._target) - 1 / self._target
        return magnitude



#%%
# Diffusion Simulation
# Base class
class DiffSim(object):
    def __init__(self, particles, bonding, diff_func, tracking=0, debug=False):
        assert self._check_input(particles, bonding, diff_func)
        self._particles = particles
        self._bonding = bonding
        self._diff_func = diff_func
        self._tracking = int(tracking)
        self._debug = debug
        self._n_iter = 0
        self._update_vec = pd.DataFrame({
            'id': self._particles['id'], 
            'dx': np.zeros(self._particles.shape[0]), 
            'dy': np.zeros(self._particles.shape[0])
        })
        self._log = {
            'iteration': [self._n_iter], 
            'particles': [self._particles], 
            'update_vec': [self._update_vec]
        }
    
    def _check_input(self, particles, bonding, diff_func):
        # Check input data type
        # particles
        if not isinstance(particles, pd.DataFrame):
            raise TypeError('`particles` must be a pd.DataFrame')
        elif set(particles.columns) != set(['id', 'x', 'y']):
            raise ValueError('`particles` must contain columns \'id\', \'x\', \'y\'')
        # bonding
        if not isinstance(bonding, pd.DataFrame):
            raise TypeError('`bonding` must be a pd.DataFrame')
        elif set(bonding.columns) != set(['id0', 'id1', 'type']):
            raise ValueError('`bonding` must contain columns \'id0\', \'id1\', \'type\'')
        # diff_func
        if not isinstance(diff_func, dict):
            raise TypeError('`particles` must be a dict')
        # Check if bonding is valid (include all possible pairs of particles)
        if (bonding['id0'] == bonding['id1']).any():
            raise ValueError('Values of `id0` and `id1` in `bonding` must be different.')
        all_pairs = self._cartesian_merge(
            particles[['id']].rename(columns={'id': 'id0'}), 
            particles[['id']].rename(columns={'id': 'id1'})
        )
        all_pairs = all_pairs.loc[all_pairs['id0'].values != all_pairs['id1'].values]
        all_pairs = all_pairs.sort_values(by=['id0', 'id1'])
        all_pairs = all_pairs.reset_index(drop=True)
        all_pairs = all_pairs.merge(bonding, on=['id0', 'id1'], how='left')
        if all_pairs['type'].isna().any(): 
            raise ValueError('`bonding` is not complete.')
        # Check if diff_func can cater for all bonding types
        for k in bonding['type'].unique():
            if k not in diff_func.keys():
                raise ValueError('`diff_func` is not complete.')
        return True
    
    @property
    def particles(self):
        return self._particles
    
    @property
    def bonding(self):
        return self._bonding
    
    @property
    def log(self):
        return self._log
    
    @staticmethod
    def _cartesian_merge(df0, df1):
        df0 = df0.copy()
        df0['dummy_key'] = 0
        df1 = df1.copy()
        df1['dummy_key'] = 0
        updated_coords = pd.merge(df0, df1, on=['dummy_key'], suffixes=['0', '1'])
        updated_coords = updated_coords.drop(columns=['dummy_key'])
        return updated_coords
    
    def _add_log(self):
        if self._tracking > 0 and (self._n_iter % self._tracking) == 0:
            if self._debug:
                print(self._n_iter)
            self._log['iteration'].append(self._n_iter)
            self._log['particles'].append(self._particles)
            self._log['update_vec'].append(self._update_vec)
    
    def _cal_update_vec(self, df):
        df['dx'] = 0
        df['dy'] = 0
        if self._debug:
            print('>>>> _cal_update_vec(): df.head()')
            print(df.head())
        update_vec = df\
            .groupby(['id0'], as_index=False)\
            .agg({'dx': np.sum, 'dy': np.sum})
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head()')
            print(update_vec.head())
        # Tidy up
        update_vec = update_vec.rename(columns={'id0': 'id'})
        update_vec = update_vec.sort_values(by=['id'])
        update_vec = update_vec.reset_index(drop=True)
        self._update_vec = update_vec
    
    def _update_coords(self):
        # Update coordinates
        updated_coords = pd.merge(self._particles, self._update_vec, on=['id'])
        if self._debug:
            print('>>>> _update_coords(): updated_coords.head()')
            print(updated_coords.head())
        updated_coords['x'] += updated_coords['dx'].values
        updated_coords['y'] += updated_coords['dy'].values
        # Tidy up
        updated_coords = updated_coords.sort_values(by=['id'])
        updated_coords = updated_coords.reset_index(drop=True)
        self._particles = updated_coords[['id', 'x', 'y']]
    
    def sim1(self):
        # Start iteration
        self._n_iter += 1
        # All possible pairs
        all_pairs = self._bonding
        all_pairs = all_pairs.merge(self._particles.rename(columns={'id': 'id0', 'x': 'x0', 'y': 'y0'}), on='id0')
        all_pairs = all_pairs.merge(self._particles.rename(columns={'id': 'id1', 'x': 'x1', 'y': 'y1'}), on='id1')
        # Calculate update_vec vectors
        self._cal_update_vec(all_pairs)
        # Update coordinates
        self._update_coords()
        # Log current step
        self._add_log()
    
    def sim(self, n=10):
        for _ in range(n):
            self.sim1()


# Unbounded
class DiffSimUnb(DiffSim):
    def __init__(self, particles, bonding, diff_func, tracking=0, debug=False):
        DiffSim.__init__(self, particles, bonding, diff_func, tracking, debug)
    
    def _cal_update_vec(self, df):
        vec = df.apply(
            lambda row: self._diff_func[row['type']].cal_vec(row['x0'], row['y0'], row['x1'], row['y1']), 
            axis=1
        )
        if self._debug:
            print('>>>> _cal_update_vec(): vec.head()')
            print(vec.head())
        df['dx'] = vec.str[0]
        df['dy'] = vec.str[1]
        if self._debug:
            print('>>>> _cal_update_vec(): df.head()')
            print(df.head())
        update_vec = df\
            .groupby(['id0'], as_index=False)\
            .agg({'dx': np.sum, 'dy': np.sum})
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head()')
            print(update_vec.head())
        # Tidy up
        update_vec = update_vec.rename(columns={'id0': 'id'})
        update_vec = update_vec.sort_values(by=['id'])
        update_vec = update_vec.reset_index(drop=True)
        self._update_vec = update_vec


# Inter points distance restricted
class DiffSimRP(DiffSim):
    def __init__(self, particles, bonding, diff_func, alpha=0.3, tracking=0, debug=False):
        DiffSim.__init__(self, particles, bonding, diff_func, tracking, debug)
        self._alpha = alpha
    
    def _cal_update_vec(self, df):
        vec = df.apply(
            lambda row: self._diff_func[row['type']].cal_vec(row['x0'], row['y0'], row['x1'], row['y1']), 
            axis=1
        )
        if self._debug:
            print('>>>> _cal_update_vec(): vec.head()')
            print(vec.head())
        df['v_x'] = vec.str[0]
        df['v_y'] = vec.str[1]
        # Inter points constraints
        df['tolerance'] = np.sqrt((df['x1'].values - df['x0'].values)**2 + (df['y1'].values - df['y0'].values)**2)
        df['tolerance'] *= self._alpha
        if self._debug:
            print('>>>> _cal_update_vec(): df.head()')
            print(df.shape)
            print(df.head())
        update_vec = df\
            .groupby(['id0'], as_index=False)\
            .agg({'v_x': np.sum, 'v_y': np.sum, 'tolerance': np.min})
        update_vec['v_norm'] = np.sqrt(update_vec['v_x'].values**2 + update_vec['v_y'].values**2)
        update_vec['fac1'] = np.divide(
            update_vec['tolerance'].values, 
            update_vec['v_norm'].values, 
            out=np.ones(update_vec.shape[0]), 
            where=update_vec['v_norm'].values != 0
        )
        update_vec['fac1'] = np.where(update_vec['fac1'].values > 1, 1, update_vec['fac1'].values)
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head() (Inter points constraints)')
            print(update_vec.shape)
            print(update_vec.head())
        # Adjustment
        update_vec['fac'] = update_vec['fac1'].values
        update_vec['dx'] = update_vec['v_x'] * update_vec['fac'].values
        update_vec['dy'] = update_vec['v_y'] * update_vec['fac'].values
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head()')
            print(update_vec.shape)
            print(update_vec.head())
        # Tidy up
        update_vec = update_vec.rename(columns={'id0': 'id'})
        update_vec = update_vec.sort_values(by=['id'])
        update_vec = update_vec.reset_index(drop=True)
        self._update_vec = update_vec
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head() (Tidy up)')
            print(update_vec.shape)
            print(update_vec.head())


# Inter points distance and boundary restricted
class DiffSimRPB(DiffSim):
    def __init__(self, particles, bonding, boundary, diff_func, alpha=0.3, tracking=0, debug=False):
        DiffSim.__init__(self, particles, bonding, diff_func, tracking, debug)
        self._boundary = boundary
        self._alpha = alpha
    
    def _distance_to_boundary(self, x, y):
        pt = Point([x, y])
        dist = self._boundary.exterior.distance(pt)
        return dist
    
    def _cal_update_vec(self, df):
        vec = df.apply(
            lambda row: self._diff_func[row['type']].cal_vec(row['x0'], row['y0'], row['x1'], row['y1']), 
            axis=1
        )
        if self._debug:
            print('>>>> _cal_update_vec(): vec.head()')
            print(vec.head())
        df['v_x'] = vec.str[0]
        df['v_y'] = vec.str[1]
        # Inter points constraints
        df['tolerance'] = np.sqrt((df['x1'].values - df['x0'].values)**2 + (df['y1'].values - df['y0'].values)**2)
        df['tolerance'] *= self._alpha
        if self._debug:
            print('>>>> _cal_update_vec(): df.head()')
            print(df.shape)
            print(df.head())
        update_vec = df\
            .groupby(['id0'], as_index=False)\
            .agg({'v_x': np.sum, 'v_y': np.sum, 'tolerance': np.min})
        update_vec['v_norm'] = np.sqrt(update_vec['v_x'].values**2 + update_vec['v_y'].values**2)
        update_vec['fac1'] = np.divide(
            update_vec['tolerance'].values, 
            update_vec['v_norm'].values, 
            out=np.ones(update_vec.shape[0]), 
            where=update_vec['v_norm'].values != 0
        )
        update_vec['fac1'] = np.where(update_vec['fac1'].values > 1, 1, update_vec['fac1'].values)
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head() (Inter points constraints)')
            print(update_vec.shape)
            print(update_vec.head())
        # Boundary constraints
        particles = self.particles.rename(columns={'id': 'id0', 'x': 'x0', 'y': 'y0'})
        update_vec = update_vec.merge(particles, on=['id0'], how='outer')
        update_vec['dist_to_bdry'] = update_vec.apply(lambda row: self._distance_to_boundary(row['x0'], row['y0']), axis=1)
        update_vec['fac2'] = np.divide(
            update_vec['dist_to_bdry'].values, 
            update_vec['v_norm'].values, 
            out=np.ones(update_vec.shape[0]), 
            where=update_vec['v_norm'].values != 0
        )
        update_vec['fac2'] = np.where(update_vec['fac2'].values > 1, 1, update_vec['fac2'].values)
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head() (Boundary constraints)')
            print(update_vec.shape)
            print(update_vec.head())
        # Adjustment
        update_vec['fac'] = update_vec[['fac1', 'fac2']].apply(min, axis=1)
        update_vec['dx'] = update_vec['v_x'] * update_vec['fac'].values
        update_vec['dy'] = update_vec['v_y'] * update_vec['fac'].values
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head() (Adjustment)')
            print(update_vec.shape)
            print(update_vec.head())
        # Tidy up
        update_vec = update_vec.rename(columns={'id0': 'id'})
        update_vec = update_vec.sort_values(by=['id'])
        update_vec = update_vec.reset_index(drop=True)
        self._update_vec = update_vec
        if self._debug:
            print('>>>> _cal_update_vec(): update_vec.head() (Tidy up)')
            print(update_vec.shape)
            print(update_vec.head())

