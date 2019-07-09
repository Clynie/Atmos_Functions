import numpy as np
from numpy import *
import netCDF4
from netCDF4 import Dataset
from math import pi,exp



def season(path, v_name, sson, step, steps=None,years=None,year=None):

	import calendar
	"""
	Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

	Parameters
	----------
	path : char_like path
		where the nc file located 
		if season is DJF, the path should be couple ;

	v_name : char_like
		the name of variability

	sson : the char_like segmentation
		only ANNUAL, MAM, JJA, SOM, DJF

	step : int
		the data step,4-times or daily data ;

	Returns
	-------
	vort : ndarray
		n-d numpy array voticity in each grid point; dimension = (nlat, nlon).

	

	"""

	DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
	
	if sson is 'DJF'   :flag = 1 ;days = DAYS[-1] + sum(DAYS[:2])
	elif sson is 'MAM' :flag = 4 ;days = sum(DAYS[(flag-1)*3-1:flag*3-1])
	elif sson is 'JJA' :flag = 3 ;days = sum(DAYS[(flag-1)*3-1:flag*3-1])
	elif sson is 'SON' :flag = 2 ;days = sum(DAYS[(flag-1)*3-1:flag*3-1])
	else :flag = 0 ;days = sum(DAYS)

	
	if years is None and year is None:
		##### THIS MEANS TO READ THE NCEP/NCAR OR ERA REANALYSIS DATA #######

		#v_name = ''.join(str(x) for x in name[0]) ;
	
	
		if sson is 'DJF' and np.size(path) is not 2:
	
			print('THE PATH SHOULD BE A COUPLE CONTAINING TWO YEARS')
			var = None
	
		elif sson is 'ANNUAL':
	
			ncfile = Dataset(''.join(str(x) for x in path[0]),'r') ;
			var	= ncfile.variables[v_name[0]] ;
	
	
		elif sson is 'DJF' and np.size(path) is 2:
	
			ncfile = Dataset(''.join(str(x) for x in path[0]),'r') ;
			var1   = np.asarray(ncfile.variables[v_name[0]][-(DAYS[11])*step:,...],dtype=np.float64) ;
			ncfile.close()
			ncfile = Dataset(''.join(str(x) for x in path[1]),'r') ;
			var2   = np.asarray(ncfile.variables[v_name[1]][:np.sum(DAYS[0:2])*step,...],dtype=np.float64) ;
			var    = np.append(var1,var2,axis=0) ; 
	
	
		else:
	
			ncfile = Dataset(''.join(str(x) for x in path[0]),'r') ;
			var    = ncfile.variables[v_name[0]][np.sum(DAYS[0:(flag-1)*3-1])*step:np.sum(DAYS[0:flag*3-1])*step,...] ;
	
	
	
		if steps is None or steps==step:
	
			data = var ;
	
		elif divmod(step,steps)[1] == 0:
	
			if np.ndim(var) == 4 :
	
				tmp = np.zeros((var.shape[0]//(step//steps), var.shape[1], var.shape[2], var.shape[3] )) ;
	
			elif np.ndim(var) == 3 :
	
				tmp = np.zeros((var.shape[0]//(step//steps), var.shape[1], var.shape[2] )) ;
	
			for dt in range(var.shape[0]//(step//steps)):
	
				#from more_itertools import chunked			
				#tmp = [sum(x) / len(x) for x in chunked(var, step/steps)]
	
				tmp[dt,...] = np.nanmean(var[dt*step:(dt+1)*step-1,...],axis=0) ;
	
			data = tmp ;
	
		else:
	
			print('STEP SHOULD LARGER THAN STEPS AND THE REMAINDER SHOULD BE 0')
			data = None

	else:
	
		
		y = np.argmin(abs(years - year)) ;
		ncfile = Dataset(path,'r') ;
		if sson is 'ANNUAL':
	
			data = ncfile.variables[v_name][y*365+calendar.leapdays(years[0],year):(y+1)*(365)+calendar.leapdays(years[0],year+1),...] ;
			if calendar.isleap(year):
				data = np.delete(data,sum(DAYS[:2]),axis=0) # MAKE SURE IT IS 365 DAYS IN A YEAR.
		else:
			data = ncfile.variables[v_name][
				(y+1)*365+calendar.leapdays(years[0],year+1)
					-sum(DAYS[-(flag-1)*3-1:]):
				(y+1)*(365)+calendar.leapdays(years[0],year+1)
					-sum(DAYS[-(flag-1)*3-1:])+days,...] ;
			

	


	return data


def deseason(path, v_name, sson, step, steps=None, years=None, year=None):

	import calendar
	
	"""
	Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

	Parameters
	----------
	path : char_like path
		where the nc file located 
		if season is DJF, the path should be couple ;

	v_name : char_like
		the name of variability

	season : the char_like segmentation
		only ANNUAL, MAM, JJA, SOM, DJF

	step : int
		the data step,4-times or daily data ;

	Returns
	-------
	vort : ndarray
		n-d numpy array voticity in each grid point; dimension = (nlat, nlon).

	

	"""


	DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
	
	if sson is 'DJF'   :flag = 1 ;days = DAYS[-1] + sum(DAYS[:2])
	elif sson is 'MAM' :flag = 4 ;days = sum(DAYS[(flag-1)*3-1:flag*3-1])
	elif sson is 'JJA' :flag = 3 ;days = sum(DAYS[(flag-1)*3-1:flag*3-1])
	elif sson is 'SON' :flag = 2 ;days = sum(DAYS[(flag-1)*3-1:flag*3-1])
	else :flag = 0 ;

	dclim = 0 ;
	mclim = 0 ;
	for yr,clmy in enumerate(years):
		
		tmp   = season(path, v_name, 'ANNUAL', step, steps=None, years=years, year=clmy) ;
		dclim = dclim + tmp/len(years) ;
		mclim = mclim + np.nanmean(tmp,axis=0)/len(years) ;
		tmp = None ;
	Fx  = np.zeros_like(dclim,dtype=complex)	
	tmp = np.fft.fft(dclim, dclim.shape[0], axis=0)
	
	Fx[1:5,...] = tmp[1:5,...]
	Fx[-4:,...] = tmp[-4:,...] ;
	tmp = None ;
	tmp = np.real(np.fft.ifft(Fx, dclim.shape[0], axis=0));
	
	if sson is 'ANNUAL':

		data = season(path, v_name, 'ANNUAL', step, steps=None, years=years, year=year) - tmp - mclim[np.newaxis,...] ;

	elif sson is 'DJF':

		tmp1 = season(path, v_name, 'ANNUAL', step, steps=None, years=years, year=year) - tmp - mclim[np.newaxis,...] ;
		tmp2 = season(path, v_name, 'ANNUAL', step, steps=None, years=years, year=year+1) - tmp - mclim[np.newaxis,...] ;

		data = np.append(tmp1[-DAYS[-1]:,...],tmp2[:sum(DAYS[:2]),...],axis=0)
	else:

					# === it should be this but for filter, the number should be odd === #
		data = (season(path, v_name, 'ANNUAL', step, steps=None, years=years, year=year) 
				- tmp - mclim[np.newaxis,...])[-sum(DAYS[-(flag-1)*3-1:]):-sum(DAYS[-(flag-2)*3-1:]),...] ;

	return data



def cvg(A, xlon, ylat, zlev=None, planet_radius=6.378e+6,n=None):

		"""
		Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

		Parameters
		----------
		xlon : sequence or array_like
				1-d numpy array of longitude (in degree) with equal spacing in ascending order; dimension = nlon.
		ylat : sequence or array_like
				1-d numpy array of latitude (in degree) with equal spacing in ascending order; dimension = nlat.
		uwnd : ndarray
				2-d or 3-d numpy array zonal wind in each grid point; dimension = (nlat, nlon).
		vwnd : ndarray
				2-d or 3-d numpy array meridional wind in each grid point; dimension = (nlat, nlon).
		planet_radius: float
				Radius of spherical planet of interest consistent with input *area*.
		omega: float
				the Coriolis parameter

		Returns
		-------
		vort : ndarray
				2-d or 3-d numpy array voticity in each grid point; dimension = (nlat, nlon).
		avort : ndarray
				2-d or 3-d numpy array voticity in each grid point; dimension = (nlat, nlon).

		

		"""
		import cmt
		nlat = A.shape[-2] ;
		nlon = A.shape[-1] ;


		[x,y] = np.meshgrid((xlon*np.pi/180.),(ylat*np.pi/180.)) ;

		dx      = np.gradient(x)[-1] ;
		dy      = np.gradient(y)[-2] ;
		R       = planet_radius*np.cos(y)

		if A.ndim ==4 and zlev is not None :

				nlev = (A.shape)[-3] ;
				dz   = np.gradient(zlev) ;

				Ax      = np.gradient(A)[-1]/(R*dx)[np.newaxis,np.newaxis,...] ;
				Ay      = np.gradient(A)[-2]/(planet_radius*dy)[np.newaxis,np.newaxis,...] ;
				Az      = np.gradient(A)[-3]/dz[np.newaxis,:,np.newaxis,np.newaxis] ;

				if n is not None :
					Az[:,:2,...] = cmt.dydx(A,zlev,-3,2)[:,:2,...] ;

		elif A.ndim ==3 and zlev is not None :

				nlev = (A.shape)[-3] ;
				dz   = np.gradient(zlev) ;

				Ax      = np.gradient(A)[-1]/(R*dx)[np.newaxis,...] ;
				Ay      = np.gradient(A)[-2]/(planet_radius*dy)[np.newaxis,...] ;
				Az      = np.gradient(A)[-3]/dz[:,np.newaxis,np.newaxis] ;

				if n is not None :
					Az[:2,...] = cmt.dydx(A,zlev,-3,2)[:2,...] ;

		else :

				Ax      = np.gradient(A)[-1]/(R*dx) ;
				Ay      = np.gradient(A)[-2]/(planet_radius*dy) ;
				Az      = None ;


		return Ax,Ay,Az





def laplacian(Ax, xlon, Ay, ylat, Az=None, zlev=None, planet_radius=6.378e+6,n=None):

		"""
		Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

		Parameters
		----------
		xlon : sequence or array_like
				1-d numpy array of longitude (in degree) with equal spacing in ascending order; dimension = nlon.
		ylat : sequence or array_like
				1-d numpy array of latitude (in degree) with equal spacing in ascending order; dimension = nlat.
		uwnd : ndarray
				2-d or 3-d numpy array zonal wind in each grid point; dimension = (nlat, nlon).
		vwnd : ndarray
				2-d or 3-d numpy array meridional wind in each grid point; dimension = (nlat, nlon).
		planet_radius: float
				Radius of spherical planet of interest consistent with input *area*.
		omega: float
				the Coriolis parameter
		n: int
				the number np.gradient replaced by np.diff in vertical axis

		Returns
		-------
		vort : ndarray
				2-d or 3-d numpy array voticity in each grid point; dimension = (nlat, nlon).
		avort : ndarray
				2-d or 3-d numpy array voticity in each grid point; dimension = (nlat, nlon).

		

		"""

		import cmt
		nlat = (Ay.shape)[-2] ;
		nlon = (Ax.shape)[-1] ;


		[x,y] = np.meshgrid((xlon*np.pi/180),(ylat*np.pi/180)) ;

		dx      = np.gradient(x)[-1] ;
		dy      = np.gradient(y)[-2] ;
		R       = planet_radius*np.cos(y)

		if Az.ndim is not None and zlev is not None :

			nlev = (Az.shape)[-3] ;
			dz   = np.gradient(zlev) ;

			Cx      = np.gradient(Ax)[-1]/(R*dx)[np.newaxis,...] ;
			Cy      = np.gradient(Ay*np.cos(y)[np.newaxis,...])[-2]/(R*dy)[np.newaxis,...] ;
			Cz      = np.gradient(Az)[-3]/dz[:,np.newaxis,np.newaxis] ;

			if n is not None :
				Cz[:2,...] = cmt.dydx(Az,zlev,-3,2)[:2,...] ;


		else :

			Cx      = np.gradient(Ax)[-1]/(R*dx)[np.newaxis,...] ;
			Cy      = np.gradient(Ay*np.cos(y)[np.newaxis,...])[-2]/(R*dy)[np.newaxis,...] ;

			Cz      = 0 ;
			

		return Cx+Cy+Cz







	

def cmpvort(xlon, uwnd, ylat, vwnd, planet_radius=6.378e+6):

	"""
	Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

	Parameters
	----------
	xlon : sequence or array_like
		1-d numpy array of longitude (in degree) with equal spacing in ascending order; dimension = nlon.
	ylat : sequence or array_like
		1-d numpy array of latitude (in degree) with equal spacing in ascending order; dimension = nlat.
	uwnd : ndarray
		2-d or 3-d numpy array zonal wind in each grid point; dimension = (nlat, nlon).
	vwnd : ndarray
		2-d or 3-d numpy array meridional wind in each grid point; dimension = (nlat, nlon).
	planet_radius: float
		Radius of spherical planet of interest consistent with input *area*.
	omega: float
		the Coriolis parameter

	Returns
	-------
	vort : ndarray
		2-d or 3-d numpy array voticity in each grid point; dimension = (nlat, nlon).
	avort : ndarray
		2-d or 3-d numpy array voticity in each grid point; dimension = (nlat, nlon).

	

	"""

	omega = 7.292e-5
	nlat = (uwnd.shape)[1] ;
	nlon = (uwnd.shape)[2] ;


	[x,y] = np.meshgrid((xlon*np.pi/180),(ylat*np.pi/180)) ;

	dy    = np.gradient(y)[0] ;
	dx    = np.gradient(x)[1] ;
	f     = 2*omega*np.sin(y) ;

	if len(uwnd.shape) is 2 and len(vwnd.shape) is 2 :

		nlev = 1 ;

		vg    = np.gradient(vwnd)[1] ;
		ug    = np.gradient(uwnd*np.cos(y))[0] ;

		VORT  = (vg/dx - ug/dy)/(planet_radius*np.cos(y)) ;
		AVORT = VORT + f ;

	else :
		nlev = (uwnd.shape)[0] ;
		VORT = np.zeros((nlev,nlat,nlon)) ;
		AVORT = np.zeros((nlev,nlat,nlon)) ;

		for i in range(nlev) :

			vg           = np.gradient(vwnd[i,:,:])[1] ;
			ug           = np.gradient(uwnd[i,:,:]*np.cos(y))[0] ;

			VORT[i,:,:]  = (vg/dx - ug/dy)/(planet_radius*np.cos(y)) ;
			AVORT[i,:,:] = VORT[i,:,:] + f ;

	VORT[np.where(np.abs(VORT[:, :, :]) > 1e5)]  = 0
	AVORT[np.where(np.abs(AVORT[:, :, :]) > 1e5)] = 0
	
	return (VORT,AVORT)



import numpy as np
from numpy import *



def dydx(A, zlev, axis=-3,n=2):

	"""
	Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

	Parameters
	----------
	zlev : sequence or array_like
		1-d numpy array of isobaric ; dimension = nlev.
	tem : ndarray
		2-d or 3-d numpy array temperature in each grid point; dimension = (nlev, nlat, nlon).
	axis: int
		the axis in array
	n: int
		the number np.gradient replaced by  np.diff

	Returns
	-------
	theta : ndarray
		2-d or 3-d numpy array potential temperature in each grid point; dimension = (nlev, nlat, nlon).
	

	"""
	var = np.zeros_like(A) ;
	if A.ndim ==4 and zlev is not None :
		var = np.gradient(A[:,:,:,:])[axis]/np.gradient(zlev)[np.newaxis,:,np.newaxis,np.newaxis] ;
		var[:,:n,...] = np.diff(A,axis=axis)[:,:n,...]/np.diff(zlev)[np.newaxis,:n,np.newaxis,np.newaxis] ;

	elif A.ndim ==3 and zlev is not None :
		var = np.gradient(A[:,:,:],axis=axis)[:,:,:]/np.gradient(zlev[:])[:,np.newaxis,np.newaxis] ;
		var[:n,...] = np.diff(A,axis=axis)[:n,...]/np.diff(zlev[:])[:n,np.newaxis,np.newaxis] ;

	



	
	return var



import numpy as np
from numpy import *



def cmptheta(zlev, t, RDGAS=287.04, CP_AIR=1.005e+3,H=None):

	"""
	Compute equivalent latitude, and optionally <...>_Q in Nakamura and Zhu (2010).

	Parameters
	----------
	zlev : sequence or array_like
		1-d numpy array of isobaric ; dimension = nlev.
	tem : ndarray
		2-d or 3-d numpy array temperature in each grid point; dimension = (nlev, nlat, nlon).
	RDGAS: float
		Radius of spherical planet of interest consistent with input *area*.
	CP_AIR: float
		the Coriolis parameter

	Returns
	-------
	theta : ndarray
		2-d or 3-d numpy array potential temperature in each grid point; dimension = (nlev, nlat, nlon).
	

	"""
	nlev = t.shape[-3]
	theta = np.zeros_like(t)
	if H is None :
		rela = 1000/zlev
	elif H is not None :
		rela = np.exp(zlev/H) ;

	if t.ndim==1 and len(t)==len(zlev): 
		theta = t*(np.power(rela,(RDGAS/CP_AIR)).squeeze()) ;

	elif t.ndim==2 and nlev==len(zlev):

		theta = t*(np.power(rela,(RDGAS/CP_AIR)).squeeze())[:,np.newaxis] ;

	elif t.ndim==3 and nlev==len(zlev):

		theta = t*(np.power(rela,(RDGAS/CP_AIR)).squeeze())[:,np.newaxis,np.newaxis] ;

	elif t.ndim==4 and nlev==len(zlev):

		theta = t*(np.power(rela,(RDGAS/CP_AIR)).squeeze())[np.newaxis,:,np.newaxis,np.newaxis] ;
	else:

		print('THE DIMENSON IS NOT MAPPING')
		theta = None


	
	return theta



def static_stability(height,area,theta,s_et=None,n_et=None,H = None):
	"""
	The function "static_stability" computes the vertical gradient (z-derivative)
	of hemispheric-averaged potential temperature, i.e. d\tilde{theta}/dz in the def-
	inition of QGPV in eq.(3) of Huang and Nakamura (2016), by central differencing.
	At the boundary, the static stability is estimated by forward/backward differen-
	cing involving two adjacent z-grid points:

		i.e. stat_n[0] = (t0_n[1]-t0_n[0])/(height[1]-height[0])
			stat_n[-1] = (t0_n[-2]-t0_n[-1])/(height[-2]-height[-1])

	Please make inquiries and report issues via Github: https://github.com/csyhuang/hn2016_falwa/issues


	Parameters
	----------
	height : sequence or array_like
		Array of z-coordinate [in Pa] with dimension = (kmax), equally spaced
	area : ndarray
		Two-dimension numpy array specifying differential areal element of each grid point;
		dimension = (nlat, nlon).
	theta : ndarray
		Matrix of potential temperature [K] with dimension (kmax,nlat,nlon) or (kmax,nlat)
	s_et : int, optional
		Index of the latitude that defines the boundary of the Southern hemispheric domain;
		initialized as nlat/2 if not input
	n_et : int, optional
		Index of the latitude that defines the boundary of the Southern hemispheric domain;
		initialized as nlat/2 if not input


	Returns
	-------
	t0_n : sequence or array_like
		Area-weighted average of potential temperature (\tilde{\theta} in HN16)
		in the Northern hemispheric domain with dimension = (kmax)
	t0_s : sequence or array_like
		Area-weighted average of potential temperature (\tilde{\theta} in HN16)
		in the Southern hemispheric domain with dimension = (kmax)
	stat_n : sequence or array_like
		Static stability (d\tilde{\theta}/dz in HN16) in the Northern hemispheric
		domain with dimension = (kmax)
	stat_s : sequence or array_like
		Static stability (d\tilde{\theta}/dz in HN16) in the Southern hemispheric
		domain with dimension = (kmax)

	"""



	nlat = theta.shape[-1]
	if s_et==None:
		s_et = nlat//2
	if n_et==None:
		n_et = nlat//2
	stat_n = np.zeros(theta.shape[0])
	stat_s = np.zeros(theta.shape[0])
	stat   = np.zeros(theta.shape[0])

	if H is None:

		if theta.ndim==4:
			zonal_mean = np.nanmean(theta,axis=-1)
		elif theta.ndim==3:
			zonal_mean = np.nanmean(theta,axis=-1)
		elif theta.ndim==2:
			zonal_mean = theta

		if area.ndim==2:
			area_zonal_mean = np.nanmean(area,axis=-1)
		elif area.ndim==1:
			area_zonal_mean = area

		csm_n_et = np.sum(area_zonal_mean[-n_et:])
		csm_s_et = np.sum(area_zonal_mean[:s_et])
		csm_et   = np.sum(area_zonal_mean)

		t0_n = np.sum(zonal_mean[:,-n_et:]*area_zonal_mean[np.newaxis,-n_et:],axis=-1)/csm_n_et ;
		t0_s = np.sum(zonal_mean[:,:s_et]*area_zonal_mean[np.newaxis,:s_et],axis=-1)/csm_s_et ;
		t0   = np.sum(zonal_mean*area_zonal_mean[np.newaxis,:],axis=-1)/csm_et ;

		try:
			stat_n = np.gradient(t0_n)/np.gradient(height) 
			stat_s = np.gradient(t0_s)/np.gradient(height) ;
			stat   = np.gradient(t0)/np.gradient(height) ;
		except ValueError:
			stat_n = np.gradient(t0_n)/np.gradient(height,edge_order=1) ;
			stat_s = np.gradient(t0_s)/np.gradient(height,edge_order=1) ;
			stat   = np.gradient(t0)/np.gradient(height,edge_order=1) ;



	else:

		if theta.ndim>=3:
			t0_n = np.nanmean(np.nanmean(theta[...,-n_et:,:],axis=-1),axis=-1)
			t0_s = np.nanmean(np.nanmean(theta[...,:s_et,:],axis=-1),axis=-1)
			t0   = np.nanmean(np.nanmean(theta,axis=-1),axis=-1)
		elif theta.ndim==2:
			t0_n = np.nanmean(theta[:,-n_et:],axis=-1)
			t0_s = np.nanmean(theta[:,:s_et],axis=-1)
			t0   = np.nanmean(theta,axis=-1)

		height = 100*1000./np.exp(height/H);
		try:
			stat_n = np.gradient(t0_n,axis=-1)/np.gradient(height) 
			stat_s = np.gradient(t0_s,axis=-1)/np.gradient(height) ;
			stat   = np.gradient(t0,axis=-1)/np.gradient(height) ;
		except ValueError:
			stat_n = np.gradient(t0_n,axis=-1)/np.gradient(height,edge_order=1) ;
			stat_s = np.gradient(t0_s,axis=-1)/np.gradient(height,edge_order=1) ;
			stat   = np.gradient(t0,axis=-1)/np.gradient(height,edge_order=1) ;

	'''
	stat_s[1:-1] = (t0_s[2:]-t0_s[:-2])/(height[2:]-height[:-2])
	stat_n[1:-1] = (t0_n[2:]-t0_n[:-2])/(height[2:]-height[:-2])
	stat[1:-1]   = (t0_s[2:]-t0_s[:-2])/(height[2:]-height[:-2])
	stat[1:-1]   = (t0[2:]-t0[:-2])/(height[2:]-height[:-2])

	stat_n[0] = (t0_n[1]-t0_n[0])/(height[1]-height[0])
	stat_n[-1] = (t0_n[-2]-t0_n[-1])/(height[-2]-height[-1])

	stat_s[0] = (t0_s[1]-t0_s[0])/(height[1]-height[0])
	stat_s[-1] = (t0_s[-2]-t0_s[-1])/(height[-2]-height[-1])

	stat[0] = (t0[1]-t0[0])/(height[1]-height[0])
	stat[-1] = (t0[-2]-t0[-1])/(height[-2]-height[-1])
	'''

	return t0_n,t0_s,stat_n,stat_s,stat












def E_vector(xlon, ylat, zlev, u, v, t, stat, OMEGA=7.292e-5, \
			RDGAS=287.04, CP_AIR=1.005e+3, planet_radius=6.378e+6, scale_height=7000.):






	def E_Core(xlon, ylat, zlev, u, v, t, stat, OMEGA=7.292e-5, \
				RDGAS=287.04, CP_AIR=1.005e+3, planet_radius=6.378e+6, scale_height=7000.):


		f = 2*OMEGA*np.sin(np.deg2rad(ylat)) ;
		clat = np.abs(np.cos(np.deg2rad(ylat))) ;
		dphi = (ylat[2]-ylat[1])*np.pi/180. ;
		dy   = planet_radius*clat*dphi ;

		tmp = (t**2)*(RDGAS*np.exp(-(RDGAS/CP_AIR)*zlev/ \
				scale_height)/scale_height/stat)[:,np.newaxis,np.newaxis] ;
		f1  = 0.5*(v**2 - u**2 - tmp)*clat[np.newaxis,:,np.newaxis] ;
		tmp = None ;

		f2   = - v*u*(clat[np.newaxis,:,np.newaxis]) ;

		f3   = f[np.newaxis,:,np.newaxis]*np.exp(-zlev[:,np.newaxis,np.newaxis]/scale_height) * \
				(v*t/stat[:,np.newaxis,np.newaxis])


		df1 = -np.gradient(f1)[-1]/(planet_radius*clat[np.newaxis,:,np.newaxis]) ;
		df2 = -np.gradient(f2*clat[np.newaxis,:,np.newaxis])[-2]/(planet_radius*clat[np.newaxis,:,np.newaxis]) ;
		df3 = -(np.exp(-zlev/scale_height)/np.gradient(zlev))[:,np.newaxis,np.newaxis]*np.gradient(f3)[-3]


		return f1, f2, f3, df1, df2, df3



	if np.ndim(u) == 4 :

		f1  = np.zeros_like(u) ;
		f2  = np.zeros_like(u) ;
		f3  = np.zeros_like(u) ;
		df1 = np.zeros_like(u) ;
		df2 = np.zeros_like(u) ;
		df3 = np.zeros_like(u) ;

		for tt in range(u.shape[0]):

			f1[tt,...], f2[tt,...], f3[tt,...], df1[tt,...], \
			df2[tt,...], df3[tt,...] = E_Core(xlon, ylat, zlev, \
											u[tt,...], v[tt,...], t[tt,...], stat[tt,...], OMEGA=7.292e-5, \
											RDGAS=287.04, CP_AIR=1.005e+3, planet_radius=6.378e+6, scale_height=7000.);

	elif np.ndim(u) == 3:

		f1, f2, f3, df1, df2, df3 = E_Core(xlon, ylat, zlev, u, v, t, stat, OMEGA=7.292e-5, \
										RDGAS=287.04, CP_AIR=1.005e+3, planet_radius=6.378e+6, scale_height=7000.);

	else:

		print("PLEASE INPUT RIGHT VARIABABILITY")


	return f1, f2, f3, df1, df2, df3






def vca(zlev, v, scale_height=7000):

	var = np.sum((np.exp(-zlev/scale_height)*np.gradient(zlev))[:,np.newaxis,np.newaxis]*\
		np.nanmean(v, axis=0), axis=0)/np.sum((np.exp(-zlev/scale_height)*\
		np.gradient(zlev)));#v = None ;


	return var


def vcaw(ylat, zlev, v, scale_height=7000):

	clat = np.abs(np.cos(np.deg2rad(ylat)))

	var = (np.sum((np.exp(-zlev/scale_height)*np.gradient(zlev))[:,np.newaxis,np.newaxis]*\
		 np.nanmean(v, axis=0), axis=0)/np.sum((np.exp(-zlev/scale_height)*\
		 np.gradient(zlev))))*clat[:,np.newaxis] ;


	return var




##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################



import datetime

def save_nc(path, lon, lat, lev, time, var, name):


	if np.ndim(name)>0:

		if lev is not None and time is not None and lon is not None :
	
			ncfile=Dataset(path,"w",format="NETCDF4")  
	
			ncfile.createDimension("timesize",len(time))  
			ncfile.createDimension("levsize",len(lev))  
			ncfile.createDimension("latsize",len(lat))  
			ncfile.createDimension("lonsize",len(lon))  

			for i in range(len(name)):
				ncfile.createVariable(name[i],np.float64,("timesize","levsize","latsize","lonsize"))  
			longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
			latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
			level      = ncfile.createVariable("lev", np.int, ('levsize'))  
			years      = ncfile.createVariable("years", np.int, ('timesize'))  
	
			for i in range(len(name)):
				print(name[i][:])
				ncfile.variables[name[i]][:]=var[i][:]
	
			level.units        = 'meters'
			latitudes.units    = 'degree_north'
			longitudes.units   = 'degree_east'
			latitudes[:]       = lat  
			longitudes[:]      = lon  
			level[:]           = lev 
			years[:]            = time
			ncfile.description = name  
			ncfile.author      = "Lynie"  
			ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
			ncfile.close()  
	

		elif (lev is not None  and lon is not None ) and (time is None):
	
			ncfile=Dataset(path,"w",format="NETCDF4")  
	
			ncfile.createDimension("levsize",len(lev))  
			ncfile.createDimension("latsize",len(lat))  
			ncfile.createDimension("lonsize",len(lon))  

			for i in range(len(name)):
				ncfile.createVariable(name[i],np.float64,("levsize","latsize","lonsize")) 
 
			longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
			latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
			level      = ncfile.createVariable("lev", np.float64, ('levsize'))  
	
			for i in range(len(name)):
				ncfile.variables[name[i]][:]=var[i][:]
	
			level.units        = 'meters'
			latitudes.units    = 'degree_north'
			longitudes.units   = 'degree_east'
			latitudes[:]       = lat  
			longitudes[:]      = lon  
			level[:]           = lev  
			ncfile.description = name  
			ncfile.author      = "Lynie"  
			ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
			ncfile.close()  
	
	else:	

		if lev is not None and time is not None and lon is not None :

			if len(time) == np.shape(var[0]):
	
				ncfile=Dataset(path,"w",format="NETCDF4")  
	
				ncfile.createDimension("timesize",len(time))  
				ncfile.createDimension("levsize",len(lev))  
				ncfile.createDimension("latsize",len(lat))  
				ncfile.createDimension("lonsize",len(lon))  
	
				ncfile.createVariable(name,np.float64,("timesize","levsize","latsize","lonsize"))  
				longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
				latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
				level      = ncfile.createVariable("lev", np.float64, ('levsize'))  
	
				ncfile.variables[name][:]=var  
	
				level.units        = 'meters'
				latitudes.units    = 'degree_north'
				longitudes.units   = 'degree_east'
				latitudes[:]       = lat  
				longitudes[:]      = lon  
				level[:]           = lev  
				ncfile.description = name  
				ncfile.author      = "Lynie"  
				ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
				ncfile.close() 

			else: 
	
				ncfile=Dataset(path,"w",format="NETCDF4")  
	
				ncfile.createDimension("yearsize",len(time))  
				ncfile.createDimension("timesize",np.shape(var)[0])  
				ncfile.createDimension("levsize",len(lev))  
				ncfile.createDimension("latsize",len(lat))  
				ncfile.createDimension("lonsize",len(lon))  
	
				ncfile.createVariable(name,np.float64,("timesize","levsize","latsize","lonsize"))  
				longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
				latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
				level      = ncfile.createVariable("lev", np.float64, ('levsize'))  
				years      = ncfile.createVariable("year", np.int, ('yearsize'))  
	
				ncfile.variables[name][:]=var  
	
				level.units        = 'meters'
				latitudes.units    = 'degree_north'
				longitudes.units   = 'degree_east'
				years.units        = 'year'
				latitudes[:]       = lat  
				longitudes[:]      = lon  
				level[:]           = lev  
				years[:]           = time  
				ncfile.description = name  
				ncfile.author      = "Lynie"  
				ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
				ncfile.close()  
	
	
		elif (lev is not None and lon is not None) and time is None :
	
			ncfile=Dataset(path,"w",format="NETCDF4")  
	
			ncfile.createDimension("levsize",len(lev))  
			ncfile.createDimension("latsize",len(lat))  
			ncfile.createDimension("lonsize",len(lon))  
	
			ncfile.createVariable(name,np.float64,("levsize","latsize","lonsize"))  
			longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
			latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
			level      = ncfile.createVariable("lev", np.float64, ('levsize'))  
	
			ncfile.variables[name][:]=var  
	
			level.units        = 'meters'
			latitudes.units    = 'degree_north'
			longitudes.units   = 'degree_east'
			latitudes[:]       = lat  
			longitudes[:]      = lon  
			level[:]           = lev  
			ncfile.description = name  
			ncfile.author      = "Lynie"  
			ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
			ncfile.close()  
	
	
		elif lev is None and (time is not None and lon is not None) :
	
			ncfile=Dataset(path,"w",format="NETCDF4")  
	
			ncfile.createDimension("timesize",len(time))  
			ncfile.createDimension("latsize",len(lat))  
			ncfile.createDimension("lonsize",len(lon))  
	
			ncfile.createVariable(name,np.float64,("timesize","latsize","lonsize"))  
			longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
			latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
	
			ncfile.variables[name][:]=var  
	
			latitudes.units    = 'degree_north'
			longitudes.units   = 'degree_east'
			latitudes[:]       = lat  
			longitudes[:]      = lon  
			ncfile.description = name  
			ncfile.author      = "Lynie"  
			ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
			ncfile.close()  
	
		elif lev is None and time is None and lon is not None:
	
			ncfile=Dataset(path,"w",format="NETCDF4")  
	
			ncfile.createDimension("latsize",len(lat))  
			ncfile.createDimension("lonsize",len(lon))  
	
			ncfile.createVariable(name,np.float64,("latsize","lonsize"))  
			longitudes = ncfile.createVariable("lon", np.float64, ('lonsize'))  
			latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
	
			ncfile.variables[name][:]=var  
	
			latitudes.units    = 'degree_north'
			longitudes.units   = 'degree_east'
			latitudes[:]       = lat  
			longitudes[:]      = lon  
			ncfile.description = name  
			ncfile.author      = "Lynie"  
			ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
			ncfile.close()  
	
	
	
	
		elif lon is None and ( lev is not None and time is not None ) :
	
			ncfile=Dataset(path,"w",format="NETCDF4")  
	
			ncfile.createDimension("timesize",len(time))  
			ncfile.createDimension("levsize",len(lev))  
			ncfile.createDimension("latsize",len(lat))  
	
			ncfile.createVariable(name,np.float64,("timesize","levsize","latsize"))  
			latitudes  = ncfile.createVariable("lat", np.float64, ('latsize'))  
			level      = ncfile.createVariable("lev", np.float64, ('levsize'))  
	
			ncfile.variables[name][:]=var  
	
			level.units        = 'meters'
			latitudes.units    = 'degree_north'
			latitudes[:]       = lat  
			level[:]           = lev 
			ncfile.description = name  
			ncfile.author      = "Lynie"  
			ncfile.createdate  = datetime.date.today().strftime("%Y-%m-%d")
			ncfile.close()  
	
	
		else:
	
			print('PLEASE INPUT RIGHT VARIABABILITY')




	return None


'''
def compute_boundary(latsin, levsout, theta_s, theta, area, points, planet_radius=Earth_radius, vgrad=None):

	from hn2016_falwa import basis

	# Part of the lower-boundary condition. in NS(2010) eq.(15)
	Input_B0 = basis.eqvlat(latsin, levsout, theta_s, area, points, planet_radius=Earth_radius, vgrad=None) ;
	tmp_B1   = basis.eqvlat(latsin, levsout, theta, area, points, planet_radius=Earth_radius, vgrad=None) ;





	return
'''


def ERA_Mean(Dir_path, years, season, step=None, steps=None):


	import numpy as np 
	import scipy.io as sio 
	import math,os,sys,gc
	from scipy import interpolate
	from scipy.interpolate import interpn
	import calendar
	import cmt
	
	
	# read in data on lat\lon grid.
	data    = sio.loadmat('PARAM_OF_ERA_Interim.mat') ;
	lonsin  = data['lon'].squeeze() ;
	nlon    = len(lonsin) ;
	
	latsin  = data['lat'][::-1].squeeze() ;
	clat    = np.abs(np.cos(np.deg2rad(latsin)))  
	nlat    = len(latsin) ;
	
	levsin  = data['lev'].squeeze() ; 
	nlev    = len(levsin) ;
	H       = -7e+3*np.log(levsin/1000.) ;
	GL      = np.arange(30e+3,0-1e+3,-1e+3)
	levsout = 1000./np.exp(GL/7000);
	
	
	Earth_radius = 6.378e+6
	RDGAS        = 287.04
	CP_AIR       = 1.005e+3
	OMEGA        = 7.292e-5
	KAPPA        = RDGAS/CP_AIR ;
	DAY          = 8.64e+4;
	DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	
	
	
	Ttmp  = np.zeros((len(years),len(levsin),nlat,nlon))
	Utmp  = np.zeros_like(Ttmp)
	Vtmp   = np.zeros_like(Ttmp)
	Ztmp   = np.zeros_like(Ttmp)
	Wtmp = np.zeros_like(Ttmp)

	if not os.path.exists(Dir_path) :
		os.makedirs(Dir_path,mode=511) ;

	for y in range(len(years)):


		year = years[y] ;
	
		file_name = 'tem' ;v_name = ['t','t'] ;
		path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
		temp = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
		temp = cmt.cmptheta(levsin, temp, RDGAS=287.04, CP_AIR=1.005e+3,H=None) ;
		Ttmp[y,...] = np.nanmean(temp,axis=0) ;

		
		file_name = 'hgt' ;v_name = ['z','z'] ;
		path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
		Height  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
		Ztmp[y,...] = np.nanmean(Height,axis=0) ;

	
		file_name = 'uwnd' ;v_name = ['u','u'] ;
		path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
		ucomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
		Utmp[y,...] = np.nanmean(ucomp,axis=0) ;

		
		file_name = 'vwnd' ;v_name = ['v','v'] ;
		path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
		vcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
		Vtmp[y,...] = np.nanmean(vcomp,axis=0) ;

	
		file_name = 'omega' ;v_name = ['w','w'] ;
		path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
		wcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
		Wtmp[y,...] = np.nanmean(wcomp,axis=0) ;		


	return np.nanmean(Ttmp,axis=0), np.nanmean(Utmp,axis=0), np.nanmean(Vtmp,axis=0), np.nanmean(Wtmp,axis=0), np.nanmean(Ztmp,axis=0)






def ERA_Origin(Dir_path, year, season, step=None, steps=None):




	import numpy as np 
	import scipy.io as sio 
	import math,os,sys,gc
	from scipy import interpolate
	from scipy.interpolate import interpn
	import calendar
	import cmt
	
	
	# read in data on lat\lon grid.
	data    = sio.loadmat('PARAM_OF_ERA_Interim.mat') ;
	lonsin  = data['lon'].squeeze() ;
	nlon    = len(lonsin) ;
	
	latsin  = data['lat'][::-1].squeeze() ;
	clat    = np.abs(np.cos(np.deg2rad(latsin)))  
	nlat    = len(latsin) ;
	
	levsin  = data['lev'].squeeze() ; 
	nlev    = len(levsin) ;
	H       = -7e+3*np.log(levsin/1000.) ;
	GL      = np.arange(30e+3,0-1e+3,-1e+3)
	levsout = 1000./np.exp(GL/7000);

		
	file_name = 'tem' ;v_name = ['t','t'] ;
	path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
	temp = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	temp = cmt.cmptheta(levsin, temp, RDGAS=287.04, CP_AIR=1.005e+3,H=None) ;
	Ttmp = np.nanmean(temp,axis=0) ;

	
	file_name = 'hgt' ;v_name = ['z','z'] ;
	path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
	Height  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	Ztmp = np.nanmean(Height,axis=0) ;


	file_name = 'uwnd' ;v_name = ['u','u'] ;
	path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
	ucomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	Utmp = np.nanmean(ucomp,axis=0) ;

	
	file_name = 'vwnd' ;v_name = ['v','v'] ;
	path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
	vcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	Vtmp = np.nanmean(vcomp,axis=0) ;


	file_name = 'omega' ;v_name = ['w','w'] ;
	path   = [[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year) + '.nc'],
					[Dir_path + file_name + '/ecmwf-' + file_name + '-' + str(year+1) + '.nc']] ;
	wcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	Wtmp = np.nanmean(wcomp,axis=0) ;		


	return ucomp, vcomp, wcomp, Height, temp





def ERA_transient(Dir_path, year, season, zlev = None, step=None, steps=None):


	import numpy as np 
	import scipy.io as sio 
	import math,os,sys,gc
	from scipy import interpolate
	from scipy.interpolate import interpn
	import calendar
	import cmt
	
	
	# read in data on lat\lon grid.
	data    = sio.loadmat('PARAM_OF_ERA_Interim.mat') ;
	lonsin  = data['lon'].squeeze() ;
	nlon    = len(lonsin) ;
	
	latsin  = data['lat'][::-1].squeeze() ;
	clat    = np.abs(np.cos(np.deg2rad(latsin)))  
	nlat    = len(latsin) ;
	
	levsin  = data['lev'].squeeze() ; 
	nlev    = len(levsin) ;
	H       = -7e+3*np.log(levsin/1000.) ;
	GL      = np.arange(30e+3,0-1e+3,-1e+3)
	levsout = 1000./np.exp(GL/7000);

	if zlev is None:
			
		file_name = 'air' ;v_name = ['air.' + str(year), 'air.' + str(year+1)] ;	
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		temp = cmt.season(path, v_name, season, step, steps=steps) ;
		temp = cmt.cmptheta(levsin, temp, RDGAS=287.04, CP_AIR=1.005e+3,H=None)[:,:,::-1,:] ;
	
		
		file_name = 'hgt' ;v_name = ['hgt.' + str(year), 'hgt.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		Height  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	
		file_name = 'uwnd' ;v_name = ['uwnd.' + str(year), 'uwnd.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		ucomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
		
		file_name = 'vwnd' ;v_name = ['vwnd.' + str(year), 'vwnd.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		vcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;
	
	
		file_name = 'omega' ;v_name = ['omega.' + str(year), 'omega.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		wcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,:,::-1,:] ;

	else:


		file_name = 'air' ;v_name = ['air.' + str(year), 'air.' + str(year+1)] ;	
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		temp = cmt.season(path, v_name, season, step, steps=steps);
		temp = cmt.cmptheta(levsin, temp, RDGAS=287.04, CP_AIR=1.005e+3,H=None)[:,zlev,::-1,:] ;
	
		
		file_name = 'hgt' ;v_name = ['hgt.' + str(year), 'hgt.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		Height  = cmt.season(path, v_name, season, step, steps=steps)[:,zlev,::-1,:] ;
	
		file_name = 'uwnd' ;v_name = ['uwnd.' + str(year), 'uwnd.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		ucomp  = cmt.season(path, v_name, season, step, steps=steps)[:,zlev,::-1,:] ;
		
		file_name = 'vwnd' ;v_name = ['vwnd.' + str(year), 'vwnd.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		vcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,zlev,::-1,:] ;
	
	
		file_name = 'omega' ;v_name = ['omega.' + str(year), 'omega.' + str(year+1)] ;
		path   = [[Dir_path  + str(year) + '/'+ file_name + '.' + str(year) + '.nc'],
						[Dir_path  + str(year+1) + '/'+ file_name + '.' + str(year+1) + '.nc']] ;
		wcomp  = cmt.season(path, v_name, season, step, steps=steps)[:,zlev,::-1,:] ;


	return ucomp, vcomp, wcomp, Height, temp


