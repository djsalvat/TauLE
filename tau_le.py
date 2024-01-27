#!/usr/bin/python

from numpy import array, sqrt, dot, arange, vstack, diff, cumsum
from scipy.integrate import odeint 
from matplotlib.pyplot import plot, xlabel, ylabel, legend, title, show, figure, semilogy, subplot, grid
from collections import namedtuple

# representation of lumped element equations
'''
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| -(l_s+l_n+V_s+M_1) | V_s                        | 0.0                | 0.0                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| V_s                | -(l_r+l_n+V_s+V_r+M_2+M_3) | V_r                | 0.0                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| 0.0                | V_r                        | -(l_g+l_n+V_r+V_g) | V_g                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| 0.0                | 0.0                        | V_g                | -(l_t+l_n+V_g+C+D) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| M_1                | 0.0                        | 0.0                | 0.0                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| 0.0                | M_2                        | 0.0                | 0.0                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| 0.0                | M_3                        | 0.0                | 0.0                | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| 0.0                | 0.0                        | 0.0                | C                  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
| 0.0                | 0.0                        | 0.0                | D                  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
+--------------------+----------------------------+--------------------+--------------------+-----+-----+-----+-----+-----+
'''

# neutron height to velocity
def h_to_v(h):
    return sqrt(2.0*9.805*h)

# define some relevant heights and critical velocities
# h = 0 means guide height
h_g = 0.5       # height to the bottom of trap
h_top = 0.5     # height to the top of the trap
h_c_up = 0.5    # cleaner up
h_c_down = 0.38 # cleaner down
h_d_up = 0.50   # dagger up
h_d_1 = 0.38    # dagger position 1
h_d_2 = 0.25    # dagger position 2
h_d_3 = 0.00    # dagger position 3
v_mean   = h_to_v(0.80)         # typical velocity (subject to change)
v_max    = h_to_v(2.00)         # highest velocity (subject to change)
v_3      = h_to_v(0.88)         # height to the RHAC or M_3
v_g      = h_to_v(h_g)          # velocity to reach trap
v_top    = h_to_v(h_g+h_top)    # marginally trapped velocity
v_c_up   = h_to_v(h_g+h_c_up)   # reach the cleaner up
v_c_down = h_to_v(h_g+h_c_down) # reach the cleaner down

# same for dagger
v_d_up = h_to_v(h_g+h_d_up)
v_d_1  = h_to_v(h_g+h_d_1)
v_d_2  = h_to_v(h_g+h_d_2)
v_d_3  = h_to_v(h_g+h_d_3)

# a "state" determines anything that can move/change during run
# source, source-RH conductance, RH-guide conductance, guide-trap conductance,
# valve to RH dump detector, cleaner height, dagger height
State = namedtuple('State','I V_s V_r V_g M_2 v_c v_d')
# a "segment" is a state for a given amount of time
Segment = namedtuple('Segment','duration state')

# typical segments in a run
fill   = State(1.0,1.0,1.0,1.0,0.0,v_c_down, v_d_1 )
clean  = State(0.0,0.0,0.0,0.0,1.0,v_c_down, v_d_1 )
store  = State(0.0,0.0,0.0,0.0,1.0,v_c_up,   v_d_up)
count1 = State(0.0,0.0,0.0,0.0,1.0,v_c_up,   v_d_1 )
count2 = State(0.0,0.0,0.0,0.0,1.0,v_c_up,   v_d_2 )
count3 = State(0.0,0.0,0.0,0.0,1.0,v_c_up,   v_d_3 )

# produce a normal storage run sequence
def storage_sequence(t_store):
    return [Segment(300.0,fill),Segment(50.0,clean),Segment(t_store,store),Segment(20.0,count1),Segment(40.0,count2),Segment(60.0,count3)]

# take the input parameters, construct state- and velocity-dependent components for the ODEs
# dn/dt = A * n + b
# inputs: source strength, neutron lifetime, source inverse lifetime (up to GV),
# roundhouse inverse lifetime, guide inverse lifetime, trap inverse lifetime,
# trap inverse lifetime for marginally trapped neutrons, source-RH conductance,
# RH-guide conductance, guide-trap conductance, GV (M_1) conductance,
# RH dump detector (M_2) conductance, RH active cleaner (M_3) conductance,
# cleaner conductance, dagger conductance.
class TauLE:
    def __init__(self,segments,I,l_n,l_s,l_r,l_g,l_t,l_t_marginal,V_s,V_r,V_g,M_1,M_2,M_3,C,D):
        self.__dict__.update({k:v for k,v in locals().items() if k!='self'})
    # generate velocity dependence for the different quantities
    # for now, inverse lifetimes go like some power of velocity -- subject to exploration/change
    def _I(self,v,state):
        return state*self.I*v**2.0
    def _l_s(self,v):
        return self.l_s*v**(3.0/2.0)
    def _l_r(self,v):
        return self.l_r*v**(3.0/2.0)
    def _l_g(self,v):
        return self.l_g*v**(3.0/2.0)
    def _l_t(self,v):
        if v < v_g:     # UCN too slow to reach trap
            return 0.0
        elif v_top < v: # marginally trapped
            return self.l_t_marginal*(v-v_top)**(3.0/2.0)
        else:           # trappable UCN
            return self.l_t*(v-v_g)**(3.0/2.0)
    # effusion between volumes goes proportional to v
    # subtracting off reduction in velocity for height where necessary
    def _V_s(self,v,state):
        return state*self.V_s*v
    def _V_r(self,v,state):
        return state*self.V_r*v
    def _V_g(self,v,state):
        return state*(0.0 if v < v_g else self.V_g*(v-v_g))
    def _M_1(self,v):
        return self.M_1*v
    def _M_2(self,v,state):
        return state*self.M_2*v
    def _M_3(self,v,v_3):
        return 0.0 if v < v_3 else self.M_3*(v-v_3)
    def _C(self,v,v_c):
        return 0.0 if v < v_c else self.C*(v-v_c)
    def _D(self,v,v_d):
        return 0.0 if v < v_d else self.D*(v-v_d)
    # return righthand side of dn/dt = A*n + b
    #def rhs(self,v,state):
    def __call__(self,n,t,v,state):
        I  = self._I(v,state.I)
        ls = self._l_s(v)
        lr = self._l_r(v)
        lg = self._l_g(v)
        lt = self._l_t(v)
        Vs = self._V_s(v,state.V_s)
        Vr = self._V_r(v,state.V_r)
        Vg = self._V_g(v,state.V_g)
        M1 = self._M_1(v)
        M2 = self._M_2(v,state.M_2)
        M3 = self._M_3(v,v_3)
        cl =   self._C(v,state.v_c)
        dg =   self._D(v,state.v_d)
        A = [ #source              #RH                        #guide               #trap                   #M_1 #M_2 #M_3 #C   #D
             [-(ls+self.l_n+Vs+M1),                        Vs,                 0.0,                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                  Vs,-(lr+self.l_n+Vs+Vr+M2+M3),                  Vr,                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                 0.0,                        Vr,-(lg+self.l_n+Vr+Vg),                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                 0.0,                       0.0,                  Vg,-(lt+self.l_n+Vg+cl+dg), 0.0, 0.0, 0.0, 0.0, 0.0],
             [                  M1,                       0.0,                 0.0,                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                 0.0,                        M2,                 0.0,                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                 0.0,                        M3,                 0.0,                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                 0.0,                       0.0,                 0.0,                     cl, 0.0, 0.0, 0.0, 0.0, 0.0],
             [                 0.0,                       0.0,                 0.0,                     dg, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        b = [                    I,                       0.0,                 0.0,                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #return array(A),array(b)
        #A,b = self.rhs(v,state)
        return dot(A,n)+b
    
    def run(self,vs,dt):
        soln = [] # returns a list of solutions for each velocity
        for v in vs: # for a given velocity, solution is T x N
            n_0 = array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            soln.append(n_0) # for times T and elements N
            for segment in self.segments: # final t becomes initial condition for next segment
                ts       = arange(0.0,segment.duration+dt,dt)
                soln_    = odeint(self,n_0,ts,args=(v,segment.state))
                soln[-1] = vstack((soln[-1],soln_[1:,:]))
                n_0  = soln[-1][-1,:]
        return soln

if __name__=='__main__':

    # make a short storage run.
    # quantities "normalized" to some velocity scale.
    tau_short_run = TauLE(storage_sequence(100.0),
                            1e3,                         #I
                            1.0/877.7,                   #l_n
                            1.0/60.0/v_mean**(3.0/2.0),  #l_s
                            1.0/100.0/v_mean**(3.0/2.0), #l_r
                            1.0/30.0/v_mean**(3.0/2.0),  #l_g
                            1.0/1e8/v_g**(3.0/2.0),      #l_t
                            1.0/10.0/v_g**(3.0/2.0),     #l_t_marginal
                            0.03/v_mean,                 #V_s
                            0.03/v_mean,                 #V_r
                            0.01/v_mean,                 #V_g
                            1e-5/v_mean,                 #M_1
                            0.01/v_mean,                 #M_2
                            1.0/v_3,                     #M_3
                            1.0/v_mean,                  #cleaner
                            1.0/v_mean                   #dagger
                         )

    # pick a set of velocities
    vs_ = array([h_to_v(h) for h in arange(0.1,1.0+0.1,0.1)])

    soln_short = tau_short_run.run(vs_,1.0)

    # sum over all energies
    soln_full = sum(soln_short)

    # pull out each component
    n_source =      soln_full[:,0]
    n_rh     =      soln_full[:,1]
    n_guide  =      soln_full[:,2]
    n_trap   =      soln_full[:,3]
    r_gv     = diff(soln_full[:,4])
    r_rhd    = diff(soln_full[:,5])
    r_rhac   = diff(soln_full[:,6])
    r_clean  = diff(soln_full[:,7])
    r_dagger = diff(soln_full[:,8])

    elements = [
                    ('source',n_source),
                    ('RH',    n_rh),
                    ('guide', n_guide),
                    ('trap',  n_trap)
               ]

    monitors = [
                    ('GV',     r_gv),
                    ('RHD',    r_rhd),
                    ('RHAC',   r_rhac),
                    ('cleaner',r_clean),
               ]

    segment_times = cumsum([s.duration for s in tau_short_run.segments])

    # plot element populations
    figure(1)
    for j,(name,soln) in enumerate(elements):
        subplot(len(elements),1,j+1)
        plot(soln,c='k',lw=2)
        for t in segment_times:
            plot([t,t],[0.0,max(soln)],lw=2,ls=':',c='k')
        grid()
        ylabel(name,fontsize='xx-large')
        if j+1==len(elements):
            xlabel('time [s]',fontsize='xx-large')
    # plot monitor rates
    figure(2)
    for j,(name,m) in enumerate(monitors):
        subplot(len(monitors),1,j+1)
        plot(m,c='k',lw=2)
        for t in segment_times:
            plot([t,t],[0.0,max(m)],lw=2,ls=':',c='k')
        grid()
        ylabel(name+' [1/s]',fontsize='xx-large')
        if j+1==len(monitors):
            xlabel('time [s]',fontsize='xx-large')
    # plot dagger rate
    figure(3)
    plot(r_dagger,c='k',lw=2)
    for t in segment_times:
        plot([t,t],[0.0,max(r_dagger)],lw=2,ls=':',c='k')
    grid()
    ylabel('dagger [1/s]',fontsize='xx-large')
    xlabel('time [s]',fontsize='xx-large')

    show()
