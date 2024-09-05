## Mitotic waves 

Code for data analysis and simulations of mitotic waves in zebrafish embryos. The relevant notebooks are listed below. The rest are helpers or tests. 

### Data analysis 

`data_w_injection_maker.ipynb` and `data_wo_injection_marker.ipynb`: linear regression of cell division times to find the wave speed. \
`wave_speed.ipynb`: fitting wave speeds from different division rounds to find the percentage difference in cell cycle length between animal pole and margin. \
`tif_2_mp4.ipynb`: converting tif files to mp4 files. \
`cell_vol.ipynb`: computing cell volumes from 8-cell and 16-cell data. \
`nearest_neighbours_inj.ipynb`: looking at the time delays of nearest neighbours of an injected area. \

### Simulations of the Kuramoto Model 

`Kuramoto1D.ipynb`: simulations of the Kuramoto Model in 1D. \
`kuramoto.ipynb`: simulations of the Kuramoto Model in flat geometry in 2D. \
`kuramoto_hemisphere.ipynb`: simulations of the Kuramoto model on the surface of a hemisphere. Lattice points are computed using Fibonacci lattice followed by Delaunay triangulation to match nearest neighbours. 

