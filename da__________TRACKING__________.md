# Tracking algorithm


## Overview

The aim of the tracking algorithm is to take the full dataset of TRAPs and find spatially proximate TRAP objects at consecutive timestamps which can be identified as a single object that propagates through space and time. To achieve this in an efficient way, the algorithm iteraters chonologically through the dataset and at every iteration, it extracts one subrecord of all TRAPs at the current **snapshot A** and one subrecord of all TRAPs at the consecutive **snapshot B** and compares them. The respective subdatasets are named **A dataset** and **B dataset** and the associated TRAPs are termed **A TRAPs** and **B TRAPs** hereafter. The basic principle of the algorithm can be summarised as follows:

1) pick the A dataset and the B dataset
2) pick a TRAP in the A dataset and check if it appears under some maximal displacement $\epsilon$ in the B dataset
3) if both TRAPs match, assign the unique identifier, the so-called **origin ID** of the A TRAP to the B TRAP
4) move on to the next A TRAP and repeat this search for a matching B TRAP
5) when all A TRAPs have been checked, iterate to the next snapshot, i.e. the B dataset becomes the A dataset and the new B dataset is extracted from the overall record 
6) return to step 2 unless the current A snapshot is the penultimate snapshot

The algorithm is run across the full 20-years record of TRAPs. In the following paragraphs we highlight the relevant aspects of the procedure.


## Reduce the study domain

During its production we have applied the algorithm to different velocity products. In order to harmonise different datasets to exactly the same domain, the algorithm first shirnks the study domain by 0.125° such that it is bound by \[22.625° N, 42.375° N\] in latitudes and \[-159.875° E, -125.125° E\] in longitudes. To be precise, it removes all TRAP objects from the overall record for which TRAP cores lie on or beyond these new boundaries.


## Find a future TRAP candidate

In order to find B TRAP candidates that might represent the future version of a current A TRAP, the algorithm defines a search box around the current A TRAP and checks if any B TRAPs are located within this search box. The size of this search box is determined by the only free parameter $\epsilon$ and extends to $\pm\epsilon$° in zonal and meridional direction around the position of the current A TRAP. In case any B TRAP candidates are found, the algorithm calculates their distances to the A TRAP (distance between TRAP cores) and picks the closest B TRAP as the future version of the current A TRAP. The unsuccessful B TRAP candidates are registered within a *division memory* to enable a later analysis of TRAP seperations. However, the division memory is commented out in the current version.

The distance comparison is where the algorithm could be improved by defining a more precise selection criterion that for instance considers similarities in core attraction or curve shape between the A TRAP and the B TRAP candidates. The amount of B TRAP candidates and thus the runtime will depend on the choice of $\epsilon$ as it virtually sets a threshold for the distance a TRAP can propagate within 24 hours, i.e. a limit on the propagation speed. The search area should also be set from a square into a circle in the next version such that $\epsilon$ defines a radius. 

With the above we also define that one A TRAP can only be mapped to one single B TRAP. If in the B snapshot e.g. three TRAPs emerge around the position of the current A TRAP, only one of them is considered the evolution of the current A TRAP while the other two are considered newly emerging TRAPs. Vice versa, if in the A snapshot multiple close TRAPs occur and seem to coincide into one B TRAP, this B TRAP only gets passed on the origin ID of the closest A TRAP, using the same approach as before, just backwards checking. The other two A TRAPs have to end in the A snapshot. This operation is always performed when a current B TRAP candidate already bears an origin ID from some previous A TRAP. Hence, in such a situation only two A TRAPs will be compared for their distance to the current B TRAP. The unsuccessful A TRAP which does not pass on its origin ID will pass its origin ID into a *fusion memory* which allows to analyse TRAP fusions later. However, the fusion memory is commented out in the current version. If for a current A TRAP no close B TRAP candidates can be found at all, the A TRAP has its last occurrence in the A snapshot and will not pass on its origin ID.


## Prevent biased trajectories

As soon as an A TRAP is located within $\epsilon$° from one of the domain boundaries, a biased tracking of the future B TRAP is possible. In this situation, the $\epsilon$ search box will touch or cross the domain boundaries and may not capture the full set of potential B TRAP candidates since on or beyond the domain boundaries no TRAPs are computed at all. As a consequence, the tracking algorithm might pick a B TRAP candidate which is the closest candidate of the incomplete set of B TRAP candidates but is not really the future version of the A TRAP and thus might lead to a wrong trajectory estimation.

To prevent this, we introduce *bias circles* for which the radii indicate the distance of the A TRAP to all four boundaries of the domain. An unbiased B TRAP candidate must then lie within all four bias circles, i.e. its distance to the A TRAP must be smaller than all of the bias circle radii. This is almost always the case and can only be broken when the A TRAP is less than $\epsilon$° away from the domain boundaries. Note that only distances with respect to the TRAP core are considered since the tracking algorithm is designed to only track TRAP cores.

If the distance between the B TRAP candidate and the A TRAP is greater than any of the bias circle radii, the real B TRAP candidate might hide behind a domain boundary but cannot be seen by the algorithm since it wasn't computed at all. The instead-chosen B TRAP candidate then would be a biased one and should therefore be ignored in the tracking process. In this case, the origin ID of the current A TRAP is simply not assigned to the B TRAP candidate and instead ends in the current A snapshot. This way the algorithm mostly only shortens trajectories when they come too close to the domain boundaries but keeps the previous track and its lifetime, so lifetimes of such boundary-touching trajectories may only be underestimated if biased at all.

The bias detection can be sped up by a priori defining an $\epsilon$-domain with boundaries $\epsilon$° within the actual domain boundaries. Biases detection may then only be run if an A TRAP is located on or beyond a boundary of the $\epsilon$-domain.


## Assign a unique identifier - the `origin_ID`

Every tracked TRAP will need a unique label that identifies all its associated instances throughout different snapshots. This unique label is called origin ID as it is composed of the creation timestamp and the count number of a newly emerging TRAP. The count number is the number a TRAP receives by the counting of all TRAPs within the respective snapshot. Together with the timestamp of this snapshot it builds a unique identifier for every TRAP.

Some TRAP that emerges in the A snapshot will have no origin ID yet. But if an A TRAP has already existed in pervious snapshots, its origin ID has been set by the previous iterations. In the latter case, the origin ID of the A TRAP remains unchanged and will only be passed on to the - if existing - next B TRAP candidate. This way, some origin ID can propagate through several snapshots of the overall dataset and mark a long-living TRAP.


## The last snapshot

For the last snapshot there would exist no B dataset. This means the algorithm does not run on the last snapshot, which isn't necessary at all because TRAPs from the last snapshot have no future to propagate and be tracked into. However, one also wants all TRAPs from the last snapshot to bear an origin ID, i.e. either their own timestamp plus their count number if they emerge during the last snapshot or the origin ID from previous iterations if they originate from another TRAP. The latter is achieved by the algorithm itself, the first can be constructed for all B TRAPs that after the last time iteration have no origin ID assigned yet.


## Determine TRAP lifetimes and ages

The lifetime of a tracked TRAP is determined by counting the occurrences of its origin ID in the overall dataset. The tracking algorithm is designed in a way that origin IDs can only last over consecutive days without gaps. Thus the number of occurrences for a given origin ID describes for how many days in a row it has persisted. This lifetime is assigned to all TRAP instances of the respective origin ID. In order to assign an age to these TRAP instances, i.e. to track the aging of the given origin ID, one has to order them chronologically and assign counts. The implementation of this age assignement is more involved than illustrated here as the algorithm deals with a large number of TRAP objects and the method has to be efficient.


## Determine TRAP peak attraction

For every origin ID/trajectory, we determine the peak attraction rate along the trajectory and assign this metric to the overall dataset. This can be achieved by grouping all TRAP instances by their origin ID label and for every origin ID, determine the minimum value of the associated core attraction values. This peak attraction value is then assigned to all TRAP instances of the respective origin ID.


## Verification of the results

To date, we have only visually examined if the tacking algorithm works well. In animations of evolving TRAPs, we mostly observe smooth TRAP evolutions, barely abrupt changes in shape or position. Along with our [puplication](URL) we show an [animation](https://doi.org/10.5281/zenodo.10943729) of the tracking results for the year 2019. All other animations that we have made throughout this project usually show smooth evolutions of the tracked TRAPs, too. Further, the remarkable coincidence of TRAP translation speeds with the propagation speed of SSH eddies hints at a robust tracking by the algorithm since the tracking results are used to derive the actual translation speed. It gives reason to believe that the algorithm succeeds in finding the future version of a given TRAP within the dataset. Nevertheless, another method to verify the results would be sensible. This could for instance be done by comparing TRAP shapes of consecutive snapshots.


## Choice of $\epsilon$

We have run the algorithm for the following values of the search box parameter:
\begin{align*}
  \epsilon=0.1^{\circ}, 0.25^{\circ}, 0.5^{\circ}, 0.75^{\circ}, 1.0^{\circ}, 1.25^{\circ}, 1.5^{\circ}
\end{align*}
For all $\epsilon$ values we find persistent TRAPs with roughly comparable shaped distributions of TRAP lifetimes and hardly changing distributions from $\epsilon=0.75^{\circ}$ onwards. The maximum lifetimes increase with increasing $\epsilon$ value and result in 197, 294, 302, 321, 321, 321 and 321 days, respectively (these results stem from computations upon geostrophic velocites with no Eckman currents). One can infer that for $\epsilon\ge0.5^{\circ}$ the longest living TRAPs must imply instances with velocities between
\begin{align*}
  v_{max}(\epsilon=0.25^{\circ})&=\frac{111120m}{1^{\circ}} \times \frac{0.25^{\circ}}{86400s} \approx 0.32 \frac{m}{s}
\end{align*}

and

\begin{align*}
  v_{max}(\epsilon=0.5^{\circ})\approx 0.64 \frac{m}{s}.
\end{align*}

with $v_{max}(\epsilon)$ denoting the limit for TRAPs propagation speed under the given $\epsilon$ threshold.
This is still within range of the maximum geostrophic + Eckman current velocities that are present in the underlying velocity data.
But comparing it with TRAP translation speed this velocity range seems to be rather untypical for TRAPs translation. From a quick view one can estimate that more than $90\%$ of sampled TRAPs show translation speeds significantly below $v_{max}(\epsilon=0.25^{\circ}) \approx 0.32 \frac{m}{s}$. This gives reason to believe that this velocity range, the longer lifetimes as well as longer trajectories that we can observe for $\epsilon\ge0.5^{\circ}$ may result from 'jumps' of A TRAPs to unrealistically distant B TRAPs in the succeeding snapshot. As a consequence, parameter values of $\epsilon\ge0.5^{\circ}$ might introduce a bias to the tracking algorithm which would require some further analysis. On the other side, $v_{max}(\epsilon=0.1^{\circ}) \approx 0.13 \frac{m}{s}$ indicates that a parameter choice of $\epsilon=0.1^{\circ}$ might become too restrictive for an analysis of TRAP propagation.
For the course of the work we therefore choose a more conservative approach and base the analysis on a parameter value of $\epsilon=0.25^{\circ}$. 
