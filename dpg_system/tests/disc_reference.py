"""Rolling-disc simulation, derived with Kane's method. Validation pass.

Two independent checks that the derivation and the integrator are sound:
  1. energy is conserved (nothing dissipative was put in)
  2. steady precession obeys rate^2 = 4g/(radius*tilt), a result that
     appears nowhere in the derivation
Only then are its measurements worth anything.
"""
import numpy as np
from sympy import symbols, sin, cos, lambdify
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
                                     RigidBody, inertia, KanesMethod, dot)
from scipy.optimize import brentq

print('deriving...', flush=True)
q1, q2, q3 = dynamicsymbols('q1 q2 q3')
u1, u2, u3 = dynamicsymbols('u1 u2 u3')
r, m, g = symbols('r m g')

N = ReferenceFrame('N')
Y = N.orientnew('Y', 'Axis', [q1, N.z])
L = Y.orientnew('L', 'Axis', [q2, Y.x])
R = L.orientnew('R', 'Axis', [q3, L.y])
w_qd = R.ang_vel_in(N)
R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)
C = Point('C')
C.set_vel(N, 0)
Dmc = C.locatenew('Dmc', r * L.z)
Dmc.v2pt_theory(C, N, R)
I = inertia(L, m/4*r**2, m/2*r**2, m/4*r**2)
body = RigidBody('body', Dmc, R, m, (I, Dmc))
body.potential_energy = m * g * r * cos(q2)
kd = [dot(R.ang_vel_in(N) - w_qd, uv) for uv in L]
KM = KanesMethod(N, q_ind=[q1, q2, q3], u_ind=[u1, u2, u3], kd_eqs=kd)
# Gravity has to be an applied LOAD. Setting potential_energy only
# feeds the energy bookkeeping -- Kane's equations never see it, and
# the first run of this simulated a coin in free space.
KM.kanes_equations([body], [(Dmc, -m * g * N.z)])
rhs = KM.mass_matrix_full.LUsolve(KM.forcing_full)

st = [q1, q2, q3, u1, u2, u3]
par = [r, m, g]
f_rhs = lambdify(st + par, list(rhs), 'numpy')
# u in terms of the qdots, so initial conditions can be stated as
# precession / lean-rate / spin instead of frame components.
q1d, q2d, q3d = dynamicsymbols('q1 q2 q3', 1)
f_u = lambdify([q2, q1d, q2d, q3d],
               [dot(w_qd, L.x), dot(w_qd, L.y), dot(w_qd, L.z)], 'numpy')
ke = body.kinetic_energy(N)
f_E = lambdify(st + par, ke + body.potential_energy, 'numpy')
f_zcm = lambdify([q2, r], r * cos(q2), 'numpy')
print('derived.\n', flush=True)

RAD, MASS, GRAV = 0.012, 0.005, 9.80665


def deriv(y):
    return np.array(f_rhs(*y, RAD, MASS, GRAV), dtype=float).ravel()


def energy(y):
    return float(f_E(*y, RAD, MASS, GRAV))


def state_from(alpha, prec, lean_rate, spin):
    """alpha = tilt of the disc PLANE from the table."""
    q2v = np.pi/2 - alpha
    us = f_u(q2v, prec, lean_rate, spin)
    return np.array([0.0, q2v, 0.0, us[0], us[1], us[2]], dtype=float)


def run(y0, seconds, dt=1e-5):
    n = int(seconds/dt)
    out = np.zeros((n, 6))
    y = np.array(y0, float)
    for i in range(n):
        out[i] = y
        k1 = deriv(y); k2 = deriv(y + .5*dt*k1)
        k3 = deriv(y + .5*dt*k2); k4 = deriv(y + dt*k3)
        y = y + (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(y).all():
            return out[:i], dt
    return out, dt


print('=' * 68)
print('CHECK 1  energy conservation (nothing dissipative was derived in)')
print('=' * 68)
y0 = state_from(np.deg2rad(15.0), 60.0, 0.0, 20.0)
traj, dt = run(y0, 0.3)
E = np.array([energy(y) for y in traj[::200]])
print(f'  energy drift over {len(traj)*dt:.2f} s: '
      f'{abs(E.max()-E.min())/abs(E.mean())*100:.2e} %   '
      f'({"conserved" if abs(E.max()-E.min())/abs(E.mean()) < 1e-6 else "DRIFTING"})')

print()
print('=' * 68)
print('CHECK 2  steady precession vs  rate^2 = 4g/(radius*tilt)')
print('=' * 68)
print('  Steadiness means the lean does not accelerate: du1/dt = 0 with')
print('  the lean rate held at zero. Solve for the precession that does')
print('  that, then compare with the law.\n')


def lean_accel(alpha, prec, spin):
    y = state_from(alpha, prec, 0.0, spin)
    return deriv(y)[3]          # du1/dt -- the lean's acceleration


print(f'{"tilt deg":>9s} {"steady prec":>12s} {"law":>10s} {"error":>8s} '
      f'{"lean drift":>11s}')
for tilt_deg in (1.0, 2.0, 5.0, 10.0, 20.0):
    alpha = np.deg2rad(tilt_deg)
    law = np.sqrt(4.0*GRAV/(RAD*alpha))
    # the Euler-disc branch: the disc's own spin is small, the contact
    # races. Solve for the precession that holds the lean steady.
    try:
        prec = brentq(lambda p: lean_accel(alpha, p, 0.0),
                      0.05*law, 8.0*law, xtol=1e-8)
    except ValueError:
        print(f'{tilt_deg:9.1f}  no root bracketed')
        continue
    y0 = state_from(alpha, prec, 0.0, 0.0)
    traj, dt = run(y0, 0.15)
    alphas = np.pi/2 - traj[:, 1]
    drift = (alphas.max()-alphas.min())/alpha*100
    print(f'{tilt_deg:9.1f} {prec:12.2f} {law:10.2f} '
          f'{(prec-law)/law*100:7.1f}% {drift:10.2f}%')
