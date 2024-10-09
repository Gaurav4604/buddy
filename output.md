4.2 CONSERVATION OF MASS 111

## EXAMPLE 4.1

In isothermal liquid flows, the fluid density is typically a known constant. What are the
dependent field variables in this case? How many equations are needed for a successful mathematical description of such flows? What physical principles supply these equations?

## Solution

When the fluid’s temperature is constant and its density is a known constant, the thermal
energy of fluid elements cannot be changed by heat transfer or work because dT ¼ dy ¼ 0, so the
thermodynamic characterization of the flow is complete from knowledge of the density. Thus,
the dependent field variables are u, the fluid’s velocity (momentum per unit mass), and the
pressure, p. Here, p is not a thermodynamic variable; instead it is a normal force (per unit area)
developed between neighboring fluid particles that either causes or results from fluid-particle
acceleration, or arises from body forces. Thus, four equations are needed; one for each
component of u, and one for p. These equations are supplied by the principle of mass conservation, and three components of Newton’s second law for fluid motion (conservation of
momentum).

# 4.2 CONSERVATION OF MASS

Setting aside nuclear reactions and relativistic effects, mass is neither created nor
destroyed. Thus, individual mass elements e molecules, grains, fluid particles, etc. e
may be tracked within a flow field because they will not disappear and new elements
will not spontaneously appear. The equations representing conservation of mass in a flowing fluid are based on the principle that the mass of a specific collection of neighboring fluid
particles is constant. The volume occupied by a specific collection of fluid particles is called
a material volume V(t). Such a volume moves and deforms within a fluid flow so that it
always contains the same mass elements; none enter the volume and none leave it. This
implies that a material volume’s surface A(t), a material surface, must move at the local
fluid velocity u so that fluid particles inside V(t) remain inside and fluid particles outside
V(t) remain outside. Thus, a statement of conservation of mass for a material volume in a
flowing fluid is:

d Z

rðx; tÞdV ¼ 0; (4.1)

dt

VðtÞ

where r is the fluid density. Figure 3.20 depicts a material volume when the control surface
velocity b is equal to u. The primary concept here is equivalent to an infinitely flexible,
perfectly sealed thin-walled balloon containing fluid. The balloon’s contents play the role
of the material volume V(t) with the balloon itself defining the material surface A(t). And,
because the balloon is sealed, the total mass of fluid inside the balloon remains constant as
the balloon moves, expands, contracts, or deforms.


-----

## 112 4. CONSERVATION LAWS

Based on (4.1), the principle of mass conservation clearly constrains the fluid density. The
implications of (4.1) for the fluid velocity field may be better displayed by using Reynolds
transport theorem (3.35) with F r and b u to expand the time derivative in (4.1):
¼ ¼
Z vrðx; tÞ Z

dV þ rðx; tÞuðx; tÞ$n dA ¼ 0: (4.2)
vt

VðtÞ AðtÞ

This is a mass-balance statement between integrated density changes within V(t) and
integrated motion of its surface A(t). Although general and correct, (4.2) may be hard to
utilize in practice because the motion and evolution of V(t) and A(t) are determined by the
flow, which may be unknown.
To develop the integral equation that represents mass conservation for an arbitrarily moving
control volume V*(t) with surface A*(t), (4.2) must be modified to involve integrations over
V*(t) and A*(t). This modification is motivated by the frequent need to conserve mass within
a volume that is not a material volume, for example a stationary control volume. The first step
in this modification is to set F r in (3.35) to obtain:
¼


rðx; tÞb$n dA ¼ 0: (4.3)


Z
rðx; tÞdV �


d
dt


Z


V[�]ðtÞ


V[�]ðtÞ


vrðx; tÞ Z
dV
�
vt

A[�]ðtÞ


The second step is to choose the arbitrary control volume V*(t) to be instantaneously coincident with material volume V(t) so that at the moment of interest V(t) V*(t) and A(t) A*(t).
¼ ¼
At this coincidence moment, the (d/dt)!rdV-terms in (4.1) and (4.3) are not equal; however,
the volume integration of vr/vt in (4.2) is equal to that in (4.3) and the surface integral of ru$n
over A(t) is equal to that over A*(t):
Z vrðx; tÞ Z vrðx; tÞ Z Z

dV ¼ dV ¼ � rðx; tÞuðx; tÞ$n dA ¼ � rðx; tÞuðx; tÞ$n dA: (4.4)
vt vt

V[�]ðtÞ VðtÞ AðtÞ A[�]ðtÞ

where the middle equality follows from (4.2). The two ends of (4.4) allow the central volumeintegral term in (4.3) to be replaced by a surface integral to find:


Z
rðx; tÞdV þ


rðx; tÞðuðx; tÞ � bÞ$n dA ¼ 0; (4.5)


d
dt


Z


V[�]ðtÞ


A[�]ðtÞ


where u and b must both be observed in the same frame of reference; they are not otherwise
restricted. This is the general integral statement of conservation of mass for an arbitrarily
moving control volume. It can be specialized to stationary, steadily moving, accelerating,
or deforming control volumes by appropriate choice of b. In particular, when b u, the arbi¼
trary control volume becomes a material volume and (4.5) reduces to (4.1).
The differential equation that represents mass conservation is obtained by applying Gauss’
divergence theorem (2.30) to the surface integration in (4.2):
Z vrðx; tÞ Z Z �vrðx; tÞ �

dV þ rðx; tÞuðx; tÞ$n dA ¼ þ V$ðrðx; tÞuðx; tÞÞ dV ¼ 0: (4.6)
vt vt

VðtÞ AðtÞ VðtÞ


-----

4.2 CONSERVATION OF MASS 113

The final equality can only be possible if the integrand vanishes at every point in space. If the
integrand did not vanish at every point in space, then integrating (4.6) in a small volume
around a point where the integrand is nonzero would produce a nonzero integral. Thus,
(4.6) requires:

vrðx; tÞ
þ V$ðrðx; tÞuðx; tÞÞ ¼ 0 or; in index notation : [vr] ðruiÞ ¼ 0: (4.7)
vt vt [þ][ v]vxi

This relationship is called the continuity equation. It expresses the principle of conservation of
mass in differential form, but is insufficient for fully determining flow fields because it is a
single equation that involves two field quantities, r and u, and u is a vector with three
components.
The second term in (4.7) is the divergence of the mass-density flux ru. Such flux divergence
terms frequently arise in conservation statements and can be interpreted as the net loss at a
point due to divergence of a flux. For example, the local r will decrease with time if V$(ru) is
positive. Flux divergence terms are also called transport terms because they transfer quantities
from one region to another without making a net contribution over the entire field. When
integrated over the entire domain of interest, their contribution vanishes if there are no sources at the boundaries.
The continuity equation may alternatively be written using the definition of D/Dt (3.5) and
vðruiÞ=vxi ¼ uivr=vxi þ rvui=vxi [see (B3.6)]:


1
rðx; tÞ


D
(4.8)
Dt [r][ð][x][;][ t][Þ þ][ V][$][u][ð][x][;][ t][Þ ¼][ 0][:]


The derivative Dr/Dt is the time rate of change of fluid density following a fluid particle.
It will be zero for constant density flow where r constant throughout the flow field, and
¼
for incompressible flow where the density of fluid particles does not change but different fluid
particles may have different density:

Dr
Dt [h][ vr]vt (4.9)

[þ][ u][$][V][r][ ¼][ 0][:]

Taken together, (4.8) and (4.9) imply:

V$u ¼ 0 (4.10)

for incompressible flows. Constant density flows are a subset of incompressible flows;
r constant is a solution of (4.9) but it is not a general solution. A fluid is usually called
¼
incompressible if its density does not change with pressure. Liquids are almost incompressible. Gases are compressible, but for flow speeds less than w100 m/s (that is, for Mach
numbers <0.3) the fractional change of absolute pressure in a room temperature airflow
is small. In this and several other situations, density changes in the flow are also small
and (4.9) and (4.10) are valid.
The general form of the continuity equation (4.7) is typically required when the derivative
Dr/Dt is nonzero because of changes in the pressure, temperature, or molecular composition
of fluid particles.


-----

## 114 4. CONSERVATION LAWS

 EXAMPLE 4.2

The density in a horizontal flow u ¼ U(y, z)ex is given by r(x, t) ¼ f(x e Ut,y,z), where f(x, y, z) is
the density distribution at t ¼ 0. Is this flow incompressible?

## Solution

There are two ways to answer this question. First, consider (4.9) and evaluate Dr/Dt, letting
x ¼ x e Ut:


Dr
Dt [¼][ vr]vt [þ][ u][$][V][r][ ¼][ vr]vt [þ][ U][ vr]vx [¼][ vr]vx

Second, consider (4.10) and evaluate V$u:


vx
vt [þ][ U][ vr]vx


vx
vx [¼][ vr]vx [ð�][U][Þ þ][ U][ vr]vx [ð][1][Þ ¼][ 0][:]


V$u ¼ [v][U][ð][y][;][ z][Þ] þ 0 þ 0 ¼ 0:
vx

In both cases, the result is zero. This is an incompressible flow, but the density may vary when f is
not constant.

# 4.3 STREAM FUNCTIONS

Consider the steady form of the continuity equation (4.7):

V$ðruÞ ¼ 0: (4.11)

The divergence of the curl of any vector field is identically zero (see Exercise 2.21), so ru will
satisfy (4.11) when written as the curl of a vector potential J:

ru ¼ V � J; (4.12)

which can be specified in terms of two scalar functions: J ¼ cVj. Putting this specification
for J into (4.12) produces ru ¼ Vc � Vj, because the curl of any gradient is identically zero
(see Exercise 2.22). Furthermore, Vc is perpendicular to surfaces of constant c, and Vj is
perpendicular to surfaces of constant j, so the mass flux ru ¼ Vc � Vj will be parallel to
surfaces of constant c and constant j. Therefore, three-dimensional streamlines are the intersections of the two stream surfaces, or stream functions in a three-dimensional flow.
The situation is illustrated in Figure 4.1. Consider two members of each of the families of
the two stream functions c a, c b, j c, j d. The intersections shown as darkened lines
¼ ¼ ¼ ¼
in Figure 4.1 are the streamlines. The mass flux _m through the surface A bounded by the four
stream surfaces (shown in gray in Figure 4.1) is calculated with area element dA, normal n (as
shown), and Stokes’ theorem.
Defining the mass flux _m through A, and using Stokes’ theorem produces:

Z Z Z Z Z
m_ ¼ ru$n dA ¼ ðV � JÞ$n dA ¼ J$ds ¼ cVj$ds ¼ cdj

A A C C C


b d c a c d b a d c :
¼ ð � Þ þ ð � Þ ¼ ð � Þð � Þ


-----

