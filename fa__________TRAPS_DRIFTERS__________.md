# Drifter-TRAP pair detection


## Raw proximity pairs

The algorithm can be seen as a simplified version of the tracking algorithm by interpreting the drifters dataset as the A dataset and the TRAPs dataset as the B dataset. The idea is to iterate through every drifter position, which constitutes a row in the drifters dataset, and search in a circle of radius $\rho=0.75\ \rm{km}$ around the drifter for the closest TRAP, compute the distance to its core and to its closest curve point and copy the TRAP's origin ID along with other attributes to the current drifter row. 75 kilometres is roughly about 1.5 times the average speed radius of mesoscale eddy detections in this region which we consult from The altimetric Mesoscalle Eddy Trajectory Atlas (META3.2 DT) provided by [AVISO+ (2022)](https://doi.org/10.24400/527896/a01-2022.005.220209). 

The search is carried out from the drifter perspective because there are less drifters than TRAPs and one TRAP can attract multiple drifters at a time while a drifter should approach just one TRAP at a time. This simplifies the computation and makes it straightforward to compare drifter positions below 24-hourly frequency against daily TRAPs.

Iterating through the drifter dataset, the algorithm creates a search box that extends $\pm0.92$ degrees (which corresponds to 75 kilometres at the northern domain boundary at 42.5° N) in zonal and meridional direction around the position of the current drifter. The TRAPs dataset is then filtered for candidates of the same day for which the TRAP core is located inside the search box. The algorithm computes the distance in kilometres between drifter and TRAP core for all candidates and selects the candidate with the closest core distance. Then it checks if the respective core distance is $\leq0.75\ \rm{km}$ which turns the search box into a search circle. If true, it records the core distance, the distance to the TRAP's closest curve point, its origin ID and other metrics to the current drifter row.

At a given timestep, only one drifter can be mapped to one single TRAP. But this is not vice versa: One TRAP can be associated with multiple drifters at the same time. If no appropriate TRAP cancandidates can be found, a drifter won't be assigned any TRAP data. At this stage, we preserve all original drifter positions and can use the dataframe to count the number of drifter days in the domain. Eventually, we obtain a drifter dataset containing distance measures to a drifter's closest TRAP.


## Proximity pairs

In the next script, the algorithm filters the dataset for drifter positions that have been assigned a closest TRAP and creates a `pair_ID` to label every individual drifter-TRAP pair that evolves over consecutive time steps. The procedure takes into account that a specific drifter might approach a TRAP, leave it and return to the same TRAP after some time, given the TRAP persists, by separating this record into two different processes and generating a unique pair ID for every single encounter. This is an important feature since otherwise the retention time of drifter-TRAP pairs will be overestimated when drifter return to a TRAP at a later moment which introduces a strong bias to any further analysis. However, the downside of considering only timestep-consecutive encounters is that if drifter or TRAP detections simply have a short detection gap, it will split up the encounter into two different processes.

In a similar fashion as the TRAPs tracking algorithm, the pair algorithm then determines the liteime and age, i.e. proximity time and proximity age, for every drifter-TRAP pair ID. But in contrast to TRAP lifetimes, proximity times and ages are first indicated by hours instead of days. This allows to apply the algorithm to sub-daily drifter data in the future. But for our analysis, we use daily drifter positions and call these variables retention time and retention age as soon as we convert them from hours to days.

Moreover, for every instance of a drifter-TRAP pair, we determine the vector pointing from the TRAP core to the drifter and get its angle to the zonal axis 0° pointing Eastwards and angle increasing counter-clockwise. We provide this drifter angle in degree-space and in kilometre-space and use it in our further analysis to rotate all drifter tracks around TRAPs towards the zonal axis. This allows us to compute spatial histograms of drifter positions and average drifter velocities around TRAPs with respect to the orientation of the TRAP curve.


## Hyperbolic pair statistics

In a last script, we group all drifter rows by their pair ID in order to obtain a dataframe with one drifter-TRAP pair per row. Each pair will be assigned a range of old and new attributes like e.g. the lifetime of the associated TRAP, the TRAP age at first encounter, an indicator if measurements of the vorticity curve are available for all pair instances as well as arrays of the core attractions and quadrupole orders a pair involves. With this new dataframe we can analyse the characteristics of drifter-TRAP pairs from many perspectives and produce statistics. Since, among other things, we are looking for the conditions that cause hyperbolic drifter motion, we call this final dataframe **hyperbolic pair statistics (HPS)**.


