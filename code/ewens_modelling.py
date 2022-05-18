# In[1]:
import numpy as np
import pickle
from scipy.stats import poisson, expon
from scipy.special import iv, factorial
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams['font.size'] = '14'
plt.style.use('seaborn-whitegrid')


# In[2]:
IMAGE_DIR = '../plots/'


# In[3]:
def f_m_hat(x, t=1):
    return t * np.exp(-t*x) / (1 - np.exp(-t)) * (0 <= x) * (x <= 1)

def f_M_hat(x, t=1):
    return t * np.exp(t*x) / (np.exp(t) - 1) * (0 <= x) * (x <= 1)

def F_S(x, t=1):
    if not hasattr(x, '__iter__'):
        dom = [x]
    dom = np.array(x).copy()
    dom[dom < 0] = 0

    n = int(np.floor(dom.max()))
    min_x = int(np.floor(dom.min()))
    ar = np.arange(min_x, n+1)
    k = np.tile(np.arange(n+1), (n+1, 1))
    k[:min_x] = -1 
    k = k[np.tril_indices(n+1)]
    k = np.expand_dims(k[k >= 0], axis=1)

    mask = (dom >= ar.reshape(len(ar), 1)) * (dom < (ar.reshape(len(ar), 1) + 1))
    dom = np.repeat(dom * mask, ar + 1, axis=0)
    dom = dom - k
    dom = np.sqrt(t * dom * (dom >= 0))

    res = (dom != 0) * np.power(-dom, k) * iv(k, 2 * dom) / factorial(k)
    res = np.exp(-t) * res.sum(axis=0)
    res[x == 0] = np.exp(-t)
    if not hasattr(x, '__iter__'):
        return res[0]
    else:
        return res 

def F_S_hat(x, t=1):
    res = (F_S(x, t) - np.exp(-t)) / (1 - np.exp(-t))
    if not hasattr(x, '__iter__'):
        return res * (x >= 0)
    else:
        return res * (np.array(x) >= 0)

def f_S_hat(x, t=1):
    if not hasattr(x, '__iter__'):
        dom = [x]
    dom = np.array(x).copy()
    dom[dom < 0] = 0

    n = int(np.floor(dom.max()))
    min_x = int(np.floor(dom.min()))
    ar = np.arange(min_x, n+1)
    k = np.tile(np.arange(n+1), (n+1, 1))
    k[:min_x] = -1 
    k = k[np.tril_indices(n+1)]
    k = np.expand_dims(k[k >= 0], axis=1)

    mask = (dom >= ar.reshape(len(ar), 1)) * (dom < (ar.reshape(len(ar), 1) + 1))
    dom = np.repeat(dom * mask, ar + 1, axis=0)
    dom = dom - k
    dom = np.sqrt(t * dom * (dom >= 0))

    res = (dom != 0) * np.nan_to_num(np.power(-dom, k-2), posinf=0, neginf=0) * (k * iv(k, 2 * dom) + dom * iv(k+1, 2*dom)) / factorial(k)
    res = np.exp(-t) * t * res.sum(axis=0) / (1 - np.exp(-t))
    res[x == 0] = np.exp(-t) * t / (1 - np.exp(-t))
    if not hasattr(x, '__iter__'):
        return res[0]
    else:
        return res 


# In[4]:
# min pdf plot
plt.figure(figsize=(8, 5), dpi=120)

domain = np.linspace(0, 1, 100)

for theta in [0.5, 1, 2, 5]:
    plt.plot(domain, f_m_hat(domain, theta), label=f'$\\theta = {theta}$', linewidth=2)

plt.legend()
plt.savefig(IMAGE_DIR + 'pdf_min_hat.png', dpi=300, bbox_inches='tight')
plt.show()

# max pdf plot
plt.figure(figsize=(8, 5), dpi=120)

for theta in [0.5, 1, 2, 5]:
    plt.plot(domain, f_M_hat(domain, theta), label=f'$\\theta = {theta}$', linewidth=2)

plt.legend()
plt.savefig(IMAGE_DIR + 'pdf_max_hat.png', dpi=300, bbox_inches='tight')
plt.show()

# sum cdf plot
plt.figure(figsize=(8, 5), dpi=120)

domain = np.linspace(0, 6, 500)
domain_floor = np.floor(domain)
n_val = np.sort(np.unique(domain_floor))

for theta, color in zip([0.5, 1, 2, 5], ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
    f_val = f_S_hat(domain, theta)
    
    for n in n_val:
        label=f'$\\theta = {theta}$' if n == n_val.max() else ''
        plt.plot(domain[domain_floor == n], f_val[domain_floor == n], label=label, linewidth=2, color=color)

plt.legend()
plt.savefig(IMAGE_DIR + 'pdf_sum_hat.png', dpi=300, bbox_inches='tight')
plt.show()


# In[5]:
class ChineseRestaurantProcess:
    def __init__(self, theta, n):
        assert theta > 0, 'theta must be > 0'
        self.theta = theta
        self.n = n

    def _sample_A(self, i, rng):
        p = [1 / (self.theta + i - 1) for j in range(i)]
        p[-1] = p[-1] * self.theta
        A = rng.choice(range(1, i+1), p=p)
        return A

    def _sample_cycles(self, random_state):
        rng = np.random.default_rng(random_state)
        cycles = [[1]]
        for i in range(2, self.n+1):
            A = self._sample_A(i, rng)
            if A == i:
                cycles.append([i])
            else:
                for cycle_ind, cycle in enumerate(cycles):
                    if A in cycle:
                        ind = cycle.index(A)
                        break
                cycles[cycle_ind].insert(ind+1, i)
        
        return cycles

    def sample_permutation(self, random_state=None):
        cycles = self._sample_cycles(random_state=random_state)
                        
        perm = [0] * self.n
        for cycle in cycles:
            cycle.append(cycle[0])
            for ind in range(len(cycle)-1):
                perm[cycle[ind]-1] = cycle[ind+1]

        return perm

    def sample_fixed_points(self, random_state=None):
        cycles = self._sample_cycles(random_state=random_state)
        
        perm = [0] * self.n
        for cycle in cycles:
            if len(cycle) == 1:
                perm[cycle[0] - 1] = cycle[0]

        return perm


# In[6]:
def FixedPointsSample(n, size=1, theta=1, random_state=None, verbose=True):
    rng = np.random.default_rng(random_state)
    sampler = ChineseRestaurantProcess(theta, n)
    if verbose:
        fixed_points = Parallel(n_jobs=-1)(delayed(sampler.sample_fixed_points)(random_state=rng.integers(size**2)) for _ in tqdm(range(size)))
    else:
        fixed_points = Parallel(n_jobs=-1)(delayed(sampler.sample_fixed_points)(random_state=rng.integers(size**2)) for _ in range(size))

    return fixed_points

def FixedPointsCount(fixed_points_sample, n, gamma=1):
    fixed_points_mask = fixed_points_sample <= np.ceil(gamma * n)
    fixed_points = fixed_points_sample * fixed_points_mask
    fixed_points_count = (fixed_points > 0).sum(axis=1)
    unique, counts = np.unique(fixed_points_count, axis=0, return_counts=True)
    return unique, counts / counts.sum()

def MinFixedPoint(fixed_points_sample, n):
    min_fixed_point = np.array(fixed_points_sample).copy()
    min_fixed_point[min_fixed_point == 0] = n + 1
    min_fixed_point = min_fixed_point.min(axis=1)
    return min_fixed_point[min_fixed_point <= n] / n

def MaxFixedPoint(fixed_points_sample, n):
    max_fixed_point = np.array(fixed_points_sample).max(axis=1)
    return max_fixed_point[max_fixed_point > 0] / n

def FixedPointsSum(fixed_points_sample, n):
    fixed_points_sum = np.array(fixed_points_sample).sum(axis=1)
    return fixed_points_sum[fixed_points_sum > 0] / n

def MinSpacing(fixed_points_sample, n):
    fixed_points = np.array(fixed_points_sample)
    spacings = np.diff(np.maximum.accumulate(fixed_points, axis=1), axis=1)
    fixed_points_min = fixed_points.copy()
    fixed_points_min[fixed_points_min == 0] = n + 1
    spacings = np.c_[fixed_points_min.min(axis=1), spacings, n-fixed_points.max(axis=1)+1]
    spacings[spacings == 0] = n + 1
    min_spacings = spacings.min(axis=1)
    min_spacings[min_spacings == n + 1] = n
    return min_spacings / n

def MaxSpacing(fixed_points_sample, n):
    fixed_points = np.array(fixed_points_sample)
    spacings = np.diff(np.maximum.accumulate(fixed_points, axis=1), axis=1)
    fixed_points_min = fixed_points.copy()
    fixed_points_min[fixed_points_min == 0] = n + 1
    spacings = np.c_[fixed_points_min.min(axis=1), spacings, n-fixed_points.max(axis=1)+1]
    max_spacings = spacings.max(axis=1)
    max_spacings[max_spacings == n + 1] = n
    return max_spacings / n


# In[7]:
theta_range = [0.5, 1, 2, 5]
n_range = [50, 100, 500]
sample_size = 3000


# In[8]:
fixed_points_samples = {
    theta: {
        n: FixedPointsSample(n, size=sample_size, theta=theta, verbose=True) for n in n_range
    } for theta in theta_range
}

with open('fixed_points_samples.pkl', 'wb') as fout:
    pickle.dump(fixed_points_samples, fout)


# In[9]:
with open('fixed_points_samples.pkl', 'rb') as fin:
    fixed_points_samples = pickle.load(fin)


# In[10]:
markers = ['s', 'D', '^', 'X', '*']

for theta in theta_range:
    plt.figure(figsize=(10, 5), dpi=120)

    plt.plot(
        range(16),
        [poisson.pmf(k, mu=theta) for k in range(16)],
        marker='.', markersize=10,
        label=fr'$\mathbb{{P}}(X = k), \theta = {theta}$'
    )

    for ind, n in enumerate(n_range):
        fixed_points_sample = fixed_points_samples[theta][n]
        unique, freq = FixedPointsCount(fixed_points_sample, n=n)
        freq = np.append(freq, np.zeros(16-unique.max()))
        unique = np.append(unique, np.arange(unique.max(), 16))
        plt.plot(
            unique, freq, markersize=5,
            marker=markers[ind],
            label=fr'$\mathbb{{P}}(X_n = k), n = {n}$'
        )

    plt.plot(
        range(16),
        [poisson.pmf(k, mu=theta) for k in range(16)],
        marker='.', markersize=10, color='tab:blue'
    )

    plt.xlabel('$k$')
    

    plt.legend()
    plt.grid(visible=True)
    plt.savefig(IMAGE_DIR + f'fp_prob_theta_{theta}.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[11]:
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for theta, color in zip(theta_range, colors):
    plt.figure(figsize=(10, 5), dpi=120)
    
    domain = np.linspace(0, 1, 100)
    plt.plot(
        domain, 
        f_m_hat(domain, theta), 
        label=fr'$f_{{\widehat{{m}}}}(x), \theta = {theta}$', 
        linewidth=2, color=color
    )

    n = max(n_range)
    plt.hist(
        MinFixedPoint(fixed_points_samples[theta][n], n=n), bins=10,
        density=True, facecolor=color, edgecolor='black', alpha=0.5,
        label=f'$\widehat{{m}}_n / n, n = {n}$'
    )

    plt.xlabel('$x$')

    plt.legend()
    plt.grid(visible=True)
    plt.savefig(IMAGE_DIR + f'fp_min_theta_{theta}.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[12]:
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for theta, color in zip(theta_range, colors):
    plt.figure(figsize=(10, 5), dpi=120)
    
    domain = np.linspace(0, 1, 100)
    plt.plot(
        domain, 
        f_M_hat(domain, theta), 
        label=fr'$f_{{\widehat{{M}}}}(x), \theta = {theta}$', 
        linewidth=2, color=color
    )

    n = max(n_range)
    plt.hist(
        MaxFixedPoint(fixed_points_samples[theta][n], n=n), bins=10,
        density=True, facecolor=color, edgecolor='black', alpha=0.5,
        label=f'$\widehat{{M}}_n / n, n = {n}$'
    )

    plt.xlabel('$x$')

    plt.legend()
    plt.grid(visible=True)
    plt.savefig(IMAGE_DIR + f'fp_max_theta_{theta}.png', dpi=300, bbox_inches='tight')
    plt.show()
# In[13]:


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for theta, color in zip(theta_range, colors):
    plt.figure(figsize=(10, 5), dpi=120)
    n = max(n_range)
    
    fp_sum = FixedPointsSum(fixed_points_samples[theta][n], n=n)
    domain = np.linspace(0, fp_sum.max(), 500)
    domain_floor = np.floor(domain)
    n_val = np.sort(np.unique(domain_floor))
    f_val = f_S_hat(domain, theta)
    
    for _n in n_val:
        label=fr'$f_{{\widehat{{S}}}}(x), \theta = {theta}$' if _n == n_val.max() else ''
        plt.plot(domain[domain_floor == _n], f_val[domain_floor == _n], label=label, linewidth=2, color=color)
        
    plt.hist(
        fp_sum, bins=20,
        density=True, facecolor=color, edgecolor='black', alpha=0.5,
        label=f'$\widehat{{S}}_n / n, n = {n}$'
    )

    plt.xlabel('$x$')

    plt.legend()
    plt.grid(visible=True)
    plt.savefig(IMAGE_DIR + f'fp_sum_theta_{theta}.png', dpi=300, bbox_inches='tight')
    plt.show()
# In[14]:
def sample_delta(theta, size):
    nu = poisson(mu=theta).rvs(size=size)
    res = []
    for n in nu:
        X = expon.rvs(size=n+1)
        res.append(
            X[0] / ((n + 1) * X.sum())
        )
    return np.array(res)

def sample_Delta(theta, size):
    nu = poisson(mu=theta).rvs(size=size)
    res = []
    for n in nu:
        X = expon.rvs(size=n+1)
        res.append(
            (X / np.arange(1, n+2)).sum() / X.sum()
        )
    return np.array(res)


# In[15]:
for theta in theta_range:
    plt.figure(figsize=(10, 5), dpi=120)
    
    domain = np.linspace(0, 1, 500, endpoint=False)

    delta_val = sample_delta(theta, 20*sample_size)

    plt.plot(
        domain, [(delta_val <= x).mean() for x in domain],
        label=fr'$F^*_{{\delta}}(x), \theta = {theta}$'
    )

    for n in n_range:
        delta_n_val = np.sort(MinSpacing(fixed_points_samples[theta][n], n=n))
        f_val = np.array([(delta_n_val < x).mean() for x in domain])
        plt.step(
            domain, f_val,
            label=fr'$F^*_{{\delta_n}}(x), n = {n}$'
        )

    plt.xlabel('$x$')

    plt.legend()
    plt.grid(visible=True)
    plt.savefig(IMAGE_DIR + f'fp_spacing_min_theta_{theta}.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[16]:
for theta in theta_range:
    plt.figure(figsize=(10, 5), dpi=120)
    
    domain = np.linspace(0, 1, 500, endpoint=False)

    Delta_val = sample_Delta(theta, 20*sample_size)

    plt.plot(
        domain, [(Delta_val <= x).mean() for x in domain],
        label=fr'$F^*_{{\Delta}}(x), \theta = {theta}$'
    )

    for n in n_range:
        Delta_n_val = MaxSpacing(fixed_points_samples[theta][n], n=n)
        plt.step(
            domain, [(Delta_n_val <= x).mean() for x in domain],
            label=fr'$F^*_{{\Delta_n}}(x), n = {n}$'
        )

    plt.xlabel('$x$')

    plt.legend()
    plt.grid(visible=True)
    plt.savefig(IMAGE_DIR + f'fp_spacing_max_theta_{theta}.png', dpi=300, bbox_inches='tight')
    plt.show()