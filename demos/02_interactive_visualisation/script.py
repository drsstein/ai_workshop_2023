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
    self._clip()
    return self.s
  
  def step(self, u):
    self.s = (1-self.k) * self.s + self.k * u
    self._clip()
    return self.s
  
  def _clip(self):
    if self.bounds is None:
      return
    
    self.s = np.clip(self.s, self.bounds[0], self.bounds[1])
    
    
    
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
      self.anchor = self.anchor_start
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
  
    
def init(num_targets=8, trigger_threshold=0.2):

  agents = {
    'ui': HarmonicOscillator(N=num_targets),
    # generate target for user
    'user_target': TargetGenerator(num_targets),
    # close the loop with a first-order lag user
    'user': FirstOrderLag(conductivity=0.2, s0 = 0, bounds=[-4,4])
  }
  # filter energy for selection trigger
  agents['trigger_in'] = FirstOrderLag(conductivity=0.1, s0 = np.ones(num_targets) * agents['ui'].energy_start)
  # trigger by thresholding
  agents['trigger_out'] = lambda inputs: inputs < trigger_threshold
  
  return agents


def reset(agents):
  o = {}
  o['ui'] = agents['ui'].reset()
  o['user'] = agents['user'].reset()
  o['user_target'] = agents['user_target'].reset()
  o['trigger_in'] = agents['trigger_in'].reset()
  o['trigger_out'] = agents['trigger_out'](o['trigger_in'])
  return o


def step(agents, a):
  global user_target
  o = {}
  
  # reset UI if event has been triggered
  if a['trigger_out'].sum() > 0:
    agents['ui'].reset()
    agents['trigger_in'].reset()

  o['ui'] = agents['ui'].step(a['user'])
  o['trigger_in'] = agents['trigger_in'].step(o['ui']['debug']['energy'])
  o['trigger_out'] = agents['trigger_out'](o['trigger_in'])
  
  # resample user goal if their target has been triggered
  o['user_target'] = agents['user_target'].step(a['trigger_out'])
  o['user'] = agents['user'].step(o['ui']['x'][o['user_target']['s']])
  
  return o
  
agents = init()
o = reset(agents)