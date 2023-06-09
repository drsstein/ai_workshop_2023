import numpy as np

class FirstOrderLag:
  """"
  First order lag (vectorized).
  
  Args:
      conductivity (np.array): moving average weight of new input, in [0,1].
      s0 (np.array): initial state after reset. 
      bounds (np.array): minimum and maximum state values
  """
  
  def __init__(self, conductivity=0.5, s0=0, bounds=None):
    self.k = conductivity
    self.s0 = s0
    self.bounds = bounds
    
  def reset(self):
    self.s = np.copy(self.s0)
    self.s = self._clip(self.s)
    return {'x': self.s, 'dx': np.zeros_like(self.s)}
  
  def step(self, u):
    s = (1-self.k) * self.s + self.k * u
    s = self._clip(s)
    ds = s - self.s
    self.s = s
    return {'x': self.s, 'dx': ds}
  
  def _clip(self, s):
    if self.bounds is None:
      return s
    
    return np.clip(s, self.bounds[0], self.bounds[1])
  

class Gaussian:
  """ Gaussian agent. Adds Gaussian noise to its input. """
  def __init__(self, stdev=1., a0=0, seed=None, mode='additive'):
    self.stdev = stdev
    self.a0 = a0 # assumed starting input
    self.seed = seed
    self.mode = mode # {'additive', 'scaled'}
    
    self.rng = np.random.default_rng(self.seed)
    self.sample = self._sample_additive
    if self.mode == 'multiply':
      self.sample = self._sample_multiplicative
    
  def reset(self):
    return self.sample(self.a0)
  
  def step(self, a):
    return self.sample(a)
  
  def _sample_additive(self, a):
    return a + self.rng.normal(size=np.asarray(a).shape) * self.stdev
  
  def _sample_multiplicative(self, a):
    return a * self.rng.normal(size=np.asarray(a).shape) * self.stdev
  
  
class NoisyLag:
  """ 
  First-order lag on noisy observations. 
  Used as a model for user input. 
  """
  
  def __init__(self, noise, lag):
    self.noise = noise
    self.lag = lag
    
  def reset(self):
    # treat lag value as noisy observation of the state
    x = self.lag.reset()['x']
    x_noisy = self.noise.step(x)
    self.lag.s = x_noisy
    return x_noisy
    
  def step(self, action):
    x_noisy = self.noise.step(action)
    return self.lag.step(x_noisy)['x']
    
    
class HarmonicOscillator():
    
    def __init__(self, 
                 N=1, 
                 mass=1., 
                 spring_constant=1., 
                 damping_coefficient=0., 
                 dt=0.1, 
                 energy_start=1., 
                 energy_max=2):
      self.N = N
      self.mass_start = mass
      self.spring_start = spring_constant
      self.damping_start = damping_coefficient
      self.dt = dt # Euler step length
      self.energy_start = energy_start
      self.energy_max = energy_max
      
      self.x = None
      self.v = None
      self.anchor_start = 0
      self.anchor = self.anchor_start
      
    def frequency(self):
      return 1/(2*np.pi) * np.sqrt(self.spring / self.mass)
    
    def potential_energy(self, x=None, spring=None):
      x = self.x if x is None else x
      spring = self.spring if spring is None else spring
      return 0.5 * spring * (x - self.anchor)**2
    
    def kinetic_energy(self, v=None, mass=None):
      v = self.v if v is None else v
      mass = self.mass if mass is None else mass
      return 0.5 * mass * v**2
    
    def energy(self, x=None, v=None, spring=None, mass=None):
      return self.potential_energy(x, spring) + self.kinetic_energy(v, mass)
    
    def ddot_x(self, dx, v=None, spring=None, damping=None, mass=None):
      v = self.v if v is None else v
      spring = self.spring if spring is None else spring
      damping = self.damping if damping is None else damping
      mass = self.mass if mass is None else mass
      
      f_spring = -spring * dx
      f_friction = -damping * v
      f_total = f_spring + f_friction
      a = f_total / mass
      return a
    
    def reset(self):
      self.mass = self.mass_start * np.ones(self.N)
      self.spring = self.spring_start * np.ones(self.N)
      self.damping = self.damping_start * np.ones(self.N)
      # initialise all targets with equal energy and different phase
      phase = np.linspace(0, 2*np.pi, self.N+1)[:-1]
      e = self.energy_start # total mechanical energy
      max_x = np.sqrt(2*e/self.spring) # amplitude along x
      self.x = max_x * np.sin(phase) + self.anchor
      # note: maximum avoids numerical instability near zero in sqrt
      k = np.maximum(0, e - self.potential_energy())
      self.v = np.sqrt(2*k/self.mass) * np.sign(np.cos(phase))
      return {'x': self.x, 'v': self.v, 
              'debug': {'energy': self.energy(), 'mass': self.mass, 'spring': self.spring}
              }
        
    def step(self, action=0):
      # make hypothetical step with original parameters
      self.anchor = action
      dx = self.x - self.anchor
      a = self.ddot_x(dx)
      v = self.v + self.dt * a
      x = self.x + self.dt * v

      # boundary condition: limit energy
      e = self.energy(x=x, v=v)
      factor = np.minimum(1, self.energy_max / e)
      # re-estimate step with scaled parameters
      spring = self.spring * factor
      mass = self.mass * factor
      a = self.ddot_x(dx, spring=spring, mass=mass)
      self.v = self.v + self.dt * a
      self.x = self.x + self.dt * self.v

      return {'x': self.x, 'v': self.v, 
              'debug': {'energy': self.energy(spring=spring, mass=mass), 'mass': mass, 'spring': spring}
             }

    
class TargetGenerator():
  
  def __init__(self, num_targets):
    self.num_targets = num_targets
    
  def reset(self):
    self.s = np.random.choice(self.num_targets)
    self.s_one_hot = np.eye(self.num_targets)[self.s]
    return self.observation()
  
  def step(self, trigger_out):
    # reset if true positive
    if trigger_out.sum() > 0 and trigger_out[self.s]:
      return self.reset()
    
    return self.observation()
  
  def observation(self):
    return {'s': self.s, 's_one_hot': self.s_one_hot}
  
  
class Register:
  """ Register agent. Updates its state to action when action is not None,
      returns the last state on each step. Used to buffer user input.
  """
  
  def __init__(self, s0=0, bounds=None):
    self.s0 = s0
    self.bounds = bounds
    self.s = None # state
    
  def reset(self):
    self.s = self.s0
    self._clip()
    return self.s
    
  def step(self, action=None):
    if action is not None:
      self.s = action
      
    self._clip()
    return self.s
  
  def _clip(self):
    if self.bounds is None:
      return
    
    self.s = np.clip(self.s, self.bounds[0], self.bounds[1])
  

def init(is_user_human=False, num_targets=8, trigger_threshold=0.2):
  user_lag = FirstOrderLag(conductivity=0.1, s0 = 0, bounds=[-200,200])
  user_noise = Gaussian(stdev=20.)
  sim_user = NoisyLag(lag=user_lag, noise=user_noise)

  agents = {
    # mass corresponding to 0.5Hz oscillation and dt matching update frequency.
    'ui': HarmonicOscillator(N=num_targets, mass=0.1, dt=0.02, energy_start=3e3, energy_max=6e3),
    # generate target for user
    'user_target': TargetGenerator(num_targets),
    # close the loop with a first-order lag user
    'sim_user': sim_user,
    'human_user': Register(s0=0, bounds=[-200, 200])
  }
  # close the loop with human user
  # filter energy for selection trigger
  agents['trigger_in'] = FirstOrderLag(conductivity=0.05, s0 = np.ones(num_targets) * agents['ui'].energy_start)
  # trigger by thresholding
  agents['trigger_out'] = lambda inputs: inputs < trigger_threshold * agents['ui'].energy_start
  
  return agents


def reset(agents):
  o = {}
  o['ui'] = agents['ui'].reset()
  o['sim_user'] = agents['sim_user'].reset()
  o['human_user'] = agents['human_user'].reset()
  o['user_target'] = agents['user_target'].reset()
  o['trigger_in'] = agents['trigger_in'].reset()['x']
  o['trigger_out'] = agents['trigger_out'](o['trigger_in'])
  return o


def _step(is_user_human, agents, a):
  global user_target
  o = {}
  
  o['user_target'] = agents['user_target'].step(a['trigger_out'])
  o['human_user'] = agents['human_user'].step()
  o['sim_user'] = agents['sim_user'].step(a['ui']['x'][o['user_target']['s']])
  
  # reset UI if event has been triggered
  if a['trigger_out'].sum() > 0:
    agents['ui'].reset()
    agents['trigger_in'].reset()

  o['ui'] = agents['ui'].step(a['human_user'] if is_user_human else a['sim_user'])
  o['trigger_in'] = agents['trigger_in'].step(o['ui']['debug']['energy'])['x']
  o['trigger_out'] = agents['trigger_out'](o['trigger_in'])
  
  return o
  
is_user_human = False
agents = init(is_user_human)
o = reset(agents)
step = lambda agents, a: _step(is_user_human, agents, a)