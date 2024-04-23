# Infer the polarity of the vorticity field around a TRAP


## Overview

We draw a circle around every TRAP core and call this circle a vorticity circle or `vircle` where $\alpha$ describes the angle of circle parameterisation. We then interpolate the daily surrounding vorticity field to each point of the vorticity circle which gives us the vorticity along the circle which we simply call vorticity curve $\zeta(\alpha)$ or `vurve`.

The vorticity curve will reveal vortex polarity changes in the environment of a TRAP core and we can use it to find quadrupoles and other polarity patterns in the surroundings. Therefore, we divide the vorticity circle into 4 intervals of $\Delta\alpha=\pi/2$ and use them to filter the vorticity curve $\zeta(\alpha)$ for combinations of four vortices.


## Find furthest curve point

For every TRAP, we calculate the distance between its core and all its curve points to determine the curve point which is furthest from the core. The distance between this curve point and the core determines the radius of the vircle and the position of the furthest curve point defines the starting point of the vurve.


## Draw vircles

We draw a circle around every TRAP core with the distance between core and furthest curve point as radius. Actually we want to start the drawing at the position of the furthest curve point and then move counter-clockwise such that we can directly relate vorticity signs and their position along the TRAP curve. But as the TRAP curve can be oriented arbitrarily in space and we want to bypass a coordinate transformation, we take an alternative approach.  

- determine the vector pointing from core to the furthest curve point
- get its angle to the zonal axis 0° pointing Eastwards and angle increasing counter-clockwise, this is the phase shift
- define raw angles alpha for the circle parameteristaion where alpha=0° points Eastwards and alpha increases counter-clockwise
- shift all alpha values by the phase shift by simply adding it $\rightarrow$ the vircle/vurve will start at the furthest curve point
- parameterise the vircle with these new angles `beta` = `alpha` + `phase_shift`
- (later) interpolate the vorticity field to the points on the vircle

Note that `beta` and `alpha` in the script are angles with the zonal axis 0° pointing Eastward. $\alpha$, however, refers to the rotated system and $\alpha=0$° coincides with the vector pointing from the TRAP core to the furthest point on the TRAP curve. In later scripts where we analyse vorticity curves, we will represent  $\alpha$ with a variable called `alpha` which has nothing to do with the same-named variable in the first script. It's confusing.

When drawing vircles around TRAPs, we skip TRAPS which only consist of one curve point or are too close to the domain boundaries to interpolate the vorticity field to it.


## Interpolate background fields

We load the full vorticity field at a given timestamp and interpolate it to the  vircle points of all TRAP instances at the same snapshot. We also run the interpolation function over the empty vircle coordinate arrays of the unsuitable TRAPS which simply gives empty vorticity arrays.


## Build vircles with no background vorticities (nbv)

When looking at the raw vorticity along the circle parameterisation, we will find patterns that actually show dynamics similar to the reference quadrupole, just hidden within some background vorticity. Apart from studying the different quadrupole orders of the raw signal, it is also insightful to repeat the study for measurements with no background vorticity (`NBV`). Therefore, we remove a constant background vorticity from every vorticity curve.


## Find different vortex configurations

What we want to find are configurations of 4-vortices which we call quadrupoles. In case of the reference quadrupole, four vortices soak material perpendicular towards the TRAP core and transport it out again along the TRAP curve, and this for both sides of the TRAP. The polarity pattern of the reference quadrupole in the northern hemisphere is cyclonic, anticyclonic, cyclonic, anticyclonic ($\oplus\ominus\oplus\ominus$) starting from either end of a TRAP. The vorticity curve $\zeta(\alpha)$ should then ressemble a sine wave and we already see this in the ensemble mean of all vorticity curves. But we also wonder what other structures are hidden in this signal?

For each TRAP we can now infer the polarity pattern of the surrounding vorticity field by analysing the vorticity values along the respective vircle. Therefore, we determine the vorticity sign in every quarter of the vircle by computing the average vorticity in each quarter. Since we assert that the number of vircle points is divisible by 4, it is straightforward to define the four intervals along the vircle: $\alpha_I = [0,\pi/2)$, $\alpha_{II} = [\pi/2,\pi)$, $\alpha_{III} = [\pi, 3\pi/2)$ and $\alpha_{IV} = [3\pi/2, 2\pi)$.

With this, we can filter the vurve for 16 (2^4) combinations of four vortices around the TRAP of either cyclonic or anticyclonic rotation, plus one undefined pattern if none of these patterns can be detected. Vortex polarities are cyclonic (positive vorticity, $\oplus$) and anticyclonic (negative vorticity, $\ominus$) and the sequence of a pattern goes counterclockwise starting at the furthest point on the TRAP curve:

- $\oplus\oplus\oplus\oplus$  '++++' (pattern 01, configuration 'A', quadrupole order 2)
- $\oplus\oplus\oplus\ominus$  '+++-' (pattern 02, configuration 'E', quadrupole order 1)
- $\oplus\oplus\ominus\oplus$  '++-+' (pattern 03, configuration 'H', quadrupole order 3)
- $\oplus\oplus\ominus\ominus$  '++--' (pattern 04, configuration 'J', quadrupole order 2)
- $\oplus\ominus\oplus\oplus$  '+-++' (pattern 05, configuration 'E', quadrupole order 1)
- $\oplus\ominus\oplus\ominus$  '+-+-' (pattern 06, configuration 'B', quadrupole order 0)
- $\oplus\ominus\ominus\oplus$  '+--+' (pattern 07, configuration 'F', quadrupole order 2)
- $\oplus\ominus\ominus\ominus$  '+---' (pattern 08, configuration 'I', quadrupole order 1)
- $\ominus\oplus\oplus\oplus$  '-+++' (pattern 09, configuration 'H', quadrupole order 3)
- $\ominus\oplus\oplus\ominus$  '-++-' (pattern 10, configuration 'F', quadrupole order 2)
- $\ominus\oplus\ominus\oplus$  '-+-+' (pattern 11, configuration 'C', quadrupole order 4)
- $\ominus\oplus\ominus\ominus$  '-+--' (pattern 12, configuration 'G', quadrupole order 3)
- $\ominus\ominus\oplus\oplus$  '--++' (pattern 13, configuration 'J', quadrupole order 2)
- $\ominus\ominus\oplus\ominus$  '--+-' (pattern 14, configuration 'I', quadrupole order 1)
- $\ominus\ominus\ominus\oplus$  '---+' (pattern 15, configuration 'G', quadrupole order 3)
- $\ominus\ominus\ominus\ominus$  '----' (pattern 16, configuration 'D', quadrupole order 2)
- undefined (pattern 99, configuration 'Z', quadrupole order 99)

For easy computation, these patterns are defined relative to the furthest curve point, i.e. relative to the 'longest' branch of a TRAP. But for our analysis it does not really matter if a certain pattern starts at the shortest or the longest branch, i.e. patterns like '+++-' ($\oplus\oplus\oplus\ominus$) and '+-++' ($\oplus\ominus\oplus\oplus$) actually express the same vortex configuration, just viewed from different branches of the TRAP, i.e. rotated by 180°. Precisely, every pattern has such a duplicate that represents the same configuration, except for the four patterns '++++' ($\oplus\oplus\oplus\oplus$), '+-+-' ($\oplus\ominus\oplus\ominus$), '-+-+' ($\ominus\oplus\ominus\oplus$) and '----' ($\ominus\ominus\ominus\ominus$). These ones are invariant under rotations of 180° around the TRAP core and thus unique.

Therefore, when detecting the 16+1 individual patterns, we directly indicate to which of the 10+1 configurations a detection belongs. And we simplify it further by grouping the different configurations by their quadrupole order. The quadrupole order $q$ describes the number of vortices in the reference quadrupole (configuration B) that must change polarity in order to obtain a given pattern.

Eventually, we run the pattern detection twice, once for the raw vorticity measurements and once for the measurements with no background vorticity (NBV).

