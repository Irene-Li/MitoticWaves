import pandas as pd 
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
from matplotlib import pyplot as plt
from functools import reduce

class Processor(): 

	def __init__(self, filename, header, f2t, inject=False):
		'''
		filename: name of file 
		header: the row to read from 
		f2t: number of frames per second 
		inject: whether the .csv file includes a column for injection 
		'''
		if filename.endswith('csv'):
			self.df = pd.read_csv(filename, header=header, encoding='latin-1')
		elif filename.endswith('xls'): 
			self.df = pd.read_excel(filename, 'Position', header=1)
		self.f2t = f2t  
		self.inject = inject 

	def process(self, f_start=None, smoothing_width=400, round_number=0, z_cutoff=0):
		'''
		The function that automatically performs all the data processing. 

		f_start: set to None if you want the function to determine f_start 
		automatically 
		smoothing_width: only used when f_start is not None. 
		See "_find_round_start_frames"
		round_number: the round below which all cells are surface cells. 
		'''
		slices = self._sort_time_frames()
		self._find_division_frames(slices)

		total = np.array([s['count']-np.sum(s['z'] < z_cutoff) for s in slices])
		divs = np.array([d['count']-np.sum(d['z'] < z_cutoff) for d in self.divisions])
		if f_start is None:
			self._find_round_start_frames(total, divs, smoothing_width=smoothing_width)
		else: 
			self.f_start = f_start 
		self._sort_into_rounds(round_number=round_number)


	def get_spherical_polar(self, round_number=-1):
		'''
		Find the spherical polar coordinates of each cell position, 
		by fitting an ellipsoid for the round indicated by round_number. 
		'''

		[a, b, angle, R], r0 = self._find_axes(round_number)
		a, b, c = a*R, b*R, R

		spherical_polar = [] 
		for index in range(len(self.rounds)):
			r = self.rounds[index]['coor']
			r = r[:, :3] - r0 
			z = r[:, 2]
			
			d = np.sqrt(np.sum(r**2, axis=-1))
			phi, theta = self._get_phi_theta(r, a, b, c, angle)
			D =  np.array([a*np.cos(phi)*np.sin(theta), b*np.sin(phi)*np.sin(theta), c*np.cos(theta)])
			D = np.sqrt( D[0]**2 + D[1]**2 + D[2]**2)
			spherical_polar.append(np.stack([d, phi, theta, D]).T)
		return spherical_polar

	def label_surface_cells(self, spherical_polar, ratio_cutoff=0.8): 
		'''
		Using spherical polar coordinates provided to label all surface cells. 
		A cell is labelled as a surface cell if the ratio (distance to the centre)/(radius of ellipsoid) 
		is larger than ratio_cutoff. 
		'''
		for index in range(len(self.rounds)):
			r = self.rounds[index]['coor']
			s = self.rounds[index]['sur']
			if np.sum(s) < 4: 
				A = spherical_polar[index]
				d = A[:, 0] 
				D = A[:, -1]
				ratio = d/D 
				self.rounds[index]['sur'] = (ratio>ratio_cutoff)



	def plot_z(self, label, omega0=0, save=False):
		plt.figure(figsize=(10, 8))
		plt.rc('font', size=20)
		for index in range(len(self.rounds)):
			r = self.rounds[index]['coor']
			s = self.rounds[index]['sur']
			if self.inject: 
				inj = self.rounds[index]['inj']
			offset = (index-1)*omega0  
			
			t = r[:, 3]-offset
			plt.scatter(t, r[:, 2], c=s, s=20, cmap='PiYG', vmin=0, vmax=1)
			if self.inject: 
				if np.sum(inj) > 0: 
					t = t[inj]
					plt.scatter(t, r[:, 2][inj], s=100, facecolors='none', edgecolors='r')
			plt.axvline(x=(self.f_start[index])*self.f2t-offset, color='grey', linestyle='--')
		plt.ylabel('z')
		plt.xlabel('t')
		if save: 
			plt.savefig('Figures/{}_z.pdf'.format(label))
		plt.show() 

	def plot_rounds(self, label, surface=True, z_cutoff=5, save=False): 

		n_rounds = len(self.rounds)
		plt.rc('font', size=15)
		fig, axes = plt.subplots(n_rounds, 3, sharey='col', sharex='col', figsize=(18, 6*n_rounds))
		for index in range(n_rounds): 
			r = self.rounds[index]['coor']
			if surface:
				s = self.rounds[index]['sur']	 
				r = r[s]
			if self.inject:
				inj = self.rounds[index]['inj']
				if surface: 
					inj = inj[s]

			x, y, z, t = r.T
			t = t - (self.f_start[index])*self.f2t
			mz = (z > z_cutoff)
			y0 = np.mean(y)
			my = (y < y0)

			axes[index, 0].scatter(x[mz], y[mz], s=z[mz], c=t[mz], cmap='plasma', vmin=0, vmax=500)
			axes[index, 0].scatter(x[t==0], y[t==0], s=150, facecolors='none', edgecolors='g')
			axes[index, 0].set_xlabel('x')
			axes[index, 0].set_ylabel('y')

			axes[index, 1].scatter(x[mz&my], z[mz&my], c=t[mz&my], cmap='plasma', vmin=0, vmax=500)
			axes[index, 1].set_xlabel('x')
			axes[index, 1].set_ylabel('z')
			
			axes[index, 2].scatter(t[mz], z[mz], c='black')
			axes[index, 2].scatter(t[t==0], z[t==0], s=150, facecolors='none', edgecolors='g')
			axes[index, 2].axhline(y=z_cutoff, linestyle='--', color='darkorange' )
			axes[index, 2].set_ylabel('z')

			if self.inject:
				if np.sum(inj) > 0: 
					axes[index, 0].scatter(x[inj&mz], y[inj&mz], s=z[inj&mz]*3, facecolors='none', edgecolors='r')
					axes[index, 1].scatter(x[inj&mz&my], z[inj&mz&my], s=150, facecolors='none', edgecolors='r')
					axes[index, 2].scatter(t[inj&mz], z[inj&mz], s=150, facecolors='none', edgecolors='r')

		axes[0, 0].set_title('x-y projection')
		axes[0, 1].set_title('x-z projection for half of the sphere')
		axes[0, 2].set_title('division timing')
		axes[0, 2].set_xlim([0, 1000])
		if save: 
			plt.savefig('Figures/{}_division_timing_sur.pdf'.format(label))
		plt.show()

	def gradient_fit(self, surface=True, z_cutoff=0, t_cutoff=500, start_diff=0, end_diff=0, plot=False, label=None):
		bounds = self._find_bounds()

		def f(x, r, t): 
			origin = x[:3]
			t0 = x[3]
			m = x[4]
			relative_dist = np.sqrt(np.sum((r - origin[np.newaxis, :])**2, axis=-1))
			y = (t-t0)*m 
			return np.sum((y-relative_dist)**2 + (m*self.f2t)**2/3)

		fits = [] 
		n_rounds = len(self.rounds)
		if plot: 
			n = n_rounds+end_diff-start_diff
			fig, axes = plt.subplots(n, 1, figsize=(6, 5*n))
			colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
		for index in range(start_diff, n_rounds+end_diff):
			r = self.rounds[index]['coor']
			t = r[:, -1] - self.f_start[index]*self.f2t
			m = (r[:, 2] > z_cutoff) & (t < 500)
			if surface: 
				s = self.rounds[index]['sur']
				m = s & m
			r = r[m]
			t = r[:, -1] - self.f_start[index]*self.f2t
			r = r[:, :-1]
			
			cost = lambda x: f(x, r, t)
			x0 = [0, 0, 0]
			res = minimize(cost, [*x0, 0, 1], bounds=[*bounds, [-self.f2t/2, self.f2t/2], [0, None]])
			fits.append(res.x)

			if plot:
				x = res.x
				origin = x[:3]
				relative_dist = np.sqrt(np.sum((r - origin[np.newaxis, :])**2, axis=-1))
				axes[index].plot(t, relative_dist, 'x', color=colors[index])
				axes[index].plot(t, x[4]*(t-x[3]), color=colors[index])
				axes[index].set_ylabel(r'$|r-r_0|$')
				axes[index].set_title('round {}'.format(index))
		if plot:
			axes[-1].set_xlabel(r't')
			if label is not None:  
				plt.tight_layout()
				plt.savefig('Figures/{}_gradient_fit.pdf'.format(label))
			plt.show()

		return fits 

	def initial_round_number(self): 
		n = self.rounds[0]['coor'].shape[0]
		return np.round(np.log2(n))

	def _sort_time_frames(self):
		self.df = self.df[self.df['TrackID'].notnull()] # filter out NaN values in track ID
		X = np.array(self.df['Position X'])
		Y = np.array(self.df['Position Y'])
		Z = np.array(self.df['Position Z'])
		T = np.array(self.df['Time'])
		S = np.array(self.df['Surface or deep'])
		TrackID = np.array(self.df['TrackID'], dtype='int')	
		if self.inject: 
			I = np.array(self.df['Injected or uninjected'])

		slices = [] 
		for t in range(min(T), max(T)): 
			x = X[T==t]
			y = Y[T==t]
			z = Z[T==t]
			ids = TrackID[T==t]
			surface = np.array([s == 'Surface' for s in S[T==t]])
			c = len(ids)
			d = {'x': x, 'y': y, 'z': z, 'id': ids, 'sur':surface, 'count': c}
			if self.inject: 
				inj = (I[T==t] == 'Injected')
				d['inj'] = inj
				d['count'] -= np.sum(inj)
			slices.append(d)
		return slices 


	def _find_division_frames(self, slices): 
		self.divisions = [] 
		old_slice = slices[0]
		old_id = old_slice['id']
		for s in slices: 
			ids = s['id']
			indices = np.in1d(old_id, ids, invert=True)
			x = old_slice['x'][indices]
			y = old_slice['y'][indices]
			z = old_slice['z'][indices]
			sur = old_slice['sur'][indices]
			c = sum(indices)
			d = {'x':x, 'y':y, 'z':z, 'sur': sur, 'count':c, 'ids': old_id[indices]}
			if self.inject: 
				inj = old_slice['inj'][indices]
				d['inj'] = inj 
				d['count'] -= np.sum(inj)
			self.divisions.append(d)
			old_id = ids 
			old_slice = s

	def _find_round_start_frames(self, total, divs, smoothing_width=10):

		total_diff = total[:-1]-total[1:]
		a = ((divs[1:] > 0) & (total_diff > 0)).astype('int')
		w = int(smoothing_width)
		a = np.convolve(a, np.ones(w), 'valid')/w
		a = ((a > 0)).astype('int')
		diff = a[1:]-a[:-1]

		f_start = np.argwhere((diff>0))+1+w
		self.f_start = np.sort(f_start.flatten())

	def _sort_into_rounds(self, round_number=0): 
		'''
		Before round_number, all cells are surface cells 
		'''
		f_cutoff = self.f_start[round_number]
		index = 0
		for (i, d) in enumerate(self.divisions):
			if i < f_cutoff:
				d['sur'] = [True]*len(d['x']) # all cells are surface cells below the cutoff point
			if index < len(self.f_start) and i >= self.f_start[index]: 
				index += 1 
			d['index'] = index-1

		track_index = -1 
		self.rounds = [] 
		for (i, d) in enumerate(self.divisions): 
			index = d['index']
			if index > track_index and index < len(self.f_start): 
				track_index = index
				self.rounds.append({'coor': [], 'sur': []})
				if self.inject: 
					self.rounds[index]['inj'] = [] 

			if index >= 0: # exclude indices below the start of the first round
				if len(d['x'])>0:
					x = d['x']
					y = d['y']
					z = d['z']
					t_array =  np.full(len(x), i)*self.f2t
					self.rounds[index]['coor'] += zip(x, y, z, t_array)
					self.rounds[index]['sur'].extend(d['sur'])
					if self.inject: 
						self.rounds[index]['inj'].extend(d['inj'])

		for i in range(len(self.rounds)):
			for k in self.rounds[i].keys(): 
				self.rounds[i][k] = np.array(self.rounds[i][k])

	def _get_phi_theta(self, r, a, b, c, angle): 
		z = r[:, -1]
		r1 = (np.cos(angle), np.sin(angle), 0)
		r2 = (-np.sin(angle), np.cos(angle), 0)
		vector = np.array((np.dot(r, r1), np.dot(r, r2), z)) # vector in the coordinate of the ellipse axes
		phi = np.arctan2(vector[1]/b, vector[0]/a)
		theta = np.arctan2(np.sqrt((vector[0]/a)**2+(vector[1]/b)**2), z/c)
		return phi, theta     


	def _find_axes(self, round_index=-1):
		r = self.rounds[round_index]['coor']
		s = self.rounds[round_index]['sur']
		r = r[s]

		def f(a, r): 
			theta = a[-1]
			r1 = (np.cos(theta), np.sin(theta))
			r2 = (-np.sin(theta), np.cos(theta))
			return np.sqrt(np.dot(r[:, :2], r1)**2/a[0]**2 + np.dot(r[:, :2], r2)**2/a[1]**2+r[:, 2]**2)

		if len(r) < 10: 
			print('Two few data points. Need at least 10.')
		else: 
			x, y, z, t = r.T

			x0 = (min(x)+max(x))/2
			y0 = (min(y)+max(y))/2
			z0 = min(z)
			r0 = np.array((x0, y0, z0))[np.newaxis, :]

			cost = lambda a: np.std(f(a, r[:, :3]-r0))
			res = minimize(cost, (1, 1, np.pi/5), bounds=[(0.1, None), (0.1, None), (0, np.pi)])
			R = np.mean(f(res.x, r[:, :3]-r0))
			return [*res.x, R], r0  

	def _find_bounds(self):
		bounds = [] 

		for i in range(3): 
			find_min = lambda x, y: min(x, min(y['coor'][:, i]))
			find_max = lambda x, y: max(x, max(y['coor'][:, i])) 
			coor_min = reduce(find_min, self.rounds, np.inf)
			coor_max = reduce(find_max, self.rounds, 0)
			bounds.append([coor_min, coor_max])

		return bounds

			












