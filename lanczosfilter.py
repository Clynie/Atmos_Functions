import numpy as np


def lowpass_cosine_filter_coef(Cf,M):
	# Positive coeficients of cosine filter low-pass.
	coef = Cf*np.hstack((1,np.sin(np.pi*np.arange(1,M+1,1)*Cf)/(np.pi*np.arange(1,M+1,1)*Cf)));
	return coef


def spectral_window(coef,N):
	#Window of cosine filter in frequency space.
	Ff = np.arange(0,1+1e-3,2./N);
	window = np.zeros(len(Ff));
	for i in range(len(Ff)):
		window[i] = coef[0] + 2*sum(coef[1:]*np.cos(np.arange(1,len(coef),1)*np.pi*Ff[i]));
	return window, Ff


def spectral_filtering(x,window):
	# Filtering in frequency space is multiplication, (convolution in time space).

	if x.ndim == 1:

		Nx  = len(x);
		Cx  = np.fft.fft(x); 
		Cx  = Cx[0:divmod(Nx,2)[0]+1];
		CxH = np.zeros_like(x, dtype=complex) ; 
		CxH[:len(Cx*window)] = Cx*window;
		CxH[len(Cx*window)-1:] = np.conjugate(CxH[1:len(Cx*window)][::-1]) ;

	elif x.ndim == 3:

		Nx  = x.shape[0] ;
		Cx  = np.fft.fft(x, x.shape[0], axis=0)
		Cx  = Cx[0:divmod(Nx,2)[0]+1,...];
		CxH = np.zeros_like(x, dtype=complex) ; 

		tmp = Cx*window[:,np.newaxis,np.newaxis] ;

		CxH[:len(tmp),...] = tmp;
		CxH[len(tmp)-1:,...] = np.conjugate(CxH[1:len(tmp),...][::-1,...]) ;
		y = np.real(np.fft.ifft(CxH, x.shape[0], axis=0));


	elif x.ndim == 4:

		Nx  = x.shape[0] ;
		Cx  = np.fft.fft(x, x.shape[0], axis=0)
		Cx  = Cx[0:divmod(Nx,2)[0]+1,...];
		CxH = np.zeros_like(x, dtype=complex) ; 
		tmp = Cx*window[:,np.newaxis,np.newaxis,np.newaxis] ;

		CxH[:len(tmp),...] = tmp;
		CxH[len(tmp)-1:,...] = np.conjugate(CxH[1:len(tmp),...][::-1,...]) ;
		y = np.real(np.fft.ifft(CxH, x.shape[0], axis=0));


	return y, Cx ;


def lanczos_filter_coef(Cf,M):
	# Positive coeficients of Lanczos [low high]-pass.
	hkcs = lowpass_cosine_filter_coef(Cf,M);
	sigma = [] ;
	sigma = np.hstack((1,np.sin(np.pi*np.arange(1,M+1)/M)/(np.pi*np.arange(1,M+1)/M)));
	hkB = hkcs*sigma;
	hkA = -hkB; 
	hkA[0] = hkA[0]+1.;
	coef = [hkB,hkA];
	return coef



def lanczosfilter(x,dT,Cf,M,passes='low'):

	if passes == 'high':
		LoH = 1 ;
	else:
		LoH = 0 ;

	Nf = 1./(2*dT); # Nyquist frequency
	# Normalize the cut off frequency with the Nyquist frequency:
	Cf = Cf/Nf;
	coef = lanczos_filter_coef(Cf,M); 
	coef = coef[LoH];
	# Filter in frequency space:
	window,Ff = spectral_window(coef,len(x)); 
	Ff = Ff*Nf;
	# Filtering:
	[y,Cx] = spectral_filtering(x,window);

	return y,coef,window,Cx,Ff
