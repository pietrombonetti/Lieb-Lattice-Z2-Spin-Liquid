import numpy as np

##################################################
def find_root(f,interval,NIT=1e4,tol=1e-5,vb=False):
    a = interval[0]
    b = interval[1]
    if (f(a)*f(b)>0):
        if (vb):
            print('find_root ERROR: f(a)*f(b)>0')
            print('find_root ERROR: f(%.4f)= %.8f, f(%.4f) = %.8f'%(a,f(a),b,f(b)))
        #raise ValueError('find_root: Interval does not intersect zero')
        return np.nan
    if (abs(f(a))<tol):
        return a
    if (abs(f(b))<tol):
        return b
    for n in range(int(NIT)):
        m = (a+b)/2
        fa=f(a)
        fb=f(b)
        fm=f(m)
        if (vb):
            print('%d: m = %.5f  f(m)=%.5f | a=%.5f   b=%.5f   f(a)=%.5f   f(b)=%.5f' %(n,m,fm,a,b,fa,fb))
        if (fa*fm<0):
            b = m
        elif (fb*fm<0):
            a = m
        else:
            break 
            return m
        if (abs(a-b)/2.<tol):
            break
            
    return (a+b)/2
#######################################################################
def make_klin(Nk):
    Nk += 1
    k_lin = np.linspace(-1,1,Nk)[1:]*np.pi
    return k_lin
    
def make_H_only(Nk,pars=dict(t=1,tdd=0,tpp=0,txy=0),mud=0,mup=0):
    k_lin = make_klin(Nk)
    ky,kx = np.meshgrid(k_lin,k_lin)
    
    ckx2 = np.cos(kx/2)
    cky2 = np.cos(ky/2)
    ckx = np.cos(kx)
    cky = np.cos(ky)
    
    Hk = np.zeros(kx.shape+(3,3,),dtype=complex)
    Hk[:,:,0,1] = -2*pars['t']*ckx2
    Hk[:,:,1,0] = -2*pars['t']*np.conj(ckx2)
    Hk[:,:,0,2] = -2*pars['t']*cky2
    Hk[:,:,2,0] = -2*pars['t']*np.conj(cky2)

    Hk[:,:,0,0] = -mud - 2*pars['tdd'] * (ckx+cky)
    Hk[:,:,1,1] = -mup - 2*pars['tpp'] * ckx
    Hk[:,:,2,2] = -mup - 2*pars['tpp'] * cky
    Hk[:,:,1,2] = - 4*pars['txy'] * ckx2*cky2
    Hk[:,:,2,1] = - 4*pars['txy'] * ckx2*cky2

    return Hk

def make_H(Nk,pars=dict(t=1,tdd=0,tpp=0,txy=0),mud=0,mup=0):
    k_lin = make_klin(Nk)
    ky,kx = np.meshgrid(k_lin,k_lin)
    
    ckx2 = np.cos(kx/2)
    cky2 = np.cos(ky/2)
    ckx = np.cos(kx)
    cky = np.cos(ky)
    
    eikx = np.exp(1j*kx/2)
    eiky = np.exp(1j*ky/2)
    Uk = np.zeros(kx.shape+(3,),dtype=complex)
    Uk[:,:,0] = 1
    Uk[:,:,1] = eikx
    Uk[:,:,2] = eiky
    
    Hk = make_H_only(Nk=Nk,pars=pars,mud=mud,mup=mup)
    Hk = np.einsum('xyi,xyij,xyj->xyij',Uk,Hk,np.conj(Uk))
    
    Ek, phik = np.linalg.eigh(Hk)
    return Ek,phik, k_lin

def backBZ(q):
    if (abs(q)==np.pi):
        return np.pi
    else:
        return np.mod(q+np.pi,2*np.pi) - np.pi

def backBZ_2D(Q):
    return np.array([backBZ(Q[0]),backBZ(Q[1])])
        
def findQ(qx,qy,klin):
    qxB = backBZ(qx)
    qyB = backBZ(qy)
    iq0 = np.argmin(abs(klin))
    iqx = np.argmin(abs(qxB-klin))-iq0
    iqy = np.argmin(abs(qyB-klin))-iq0
    return iqx,iqy

def shiftQ(arr,klin,qx,qy):
    iqx,iqy = findQ(qx,qy,klin)
    ret = np.roll(arr,-iqx,axis=0)
    ret = np.roll(ret,-iqy,axis=1)
    return ret
    
####################################################################
def fermi(x,T):
    return fermi_der(x,T,n=0)

def fermi_der(x,T,n=1): 
    if (T<1e-6):
        T = 1e-6
    y = x/(2*T)
    t = np.tanh(y)
    f = 1/(2*T)
    if (n==0):
        return 0.5*(1-t)
    elif (n==1):
        return -0.5*f* (1-t**2)
    elif (n==2):
        return -0.5*f**2 * (2*t**3-2*t)
    elif (n==3):
        return -0.5*f**3 * (-6*t**4+8*t**2-2)
    else:
        print('fermi_der: n=%d order not coded' %n)
        return -1e8

def fermi_diff(arr,klin,qx,qy,T=0,tol=1e-4):
    Ek = np.repeat(arr[:,:,:,np.newaxis],arr.shape[2],axis=3)
    Ekq = shiftQ(Ek,klin,qx,qy) 
    Ekq = np.transpose(Ekq,(0,1,3,2))

    e1_e2_equal = abs(Ek-Ekq)<=tol

    ret = np.zeros(Ek.shape)
    ret += e1_e2_equal * fermi_der(Ek,T,n=1)
    ret += (1-e1_e2_equal) * (fermi(Ek,T)-fermi(Ekq,T))*Pvalue(Ek-Ekq)
    return -ret
####################################################################
def calc_chi0(energies,vectors,klin,qx,qy,T=0,tol=0,vb=False):
    FFk = fermi_diff(energies,klin,qx,qy,T,tol)
    
    uk = np.einsum('xyal,xybl->xyabl',vectors,np.conj(vectors))
    ukq = shiftQ(uk,klin,qx,qy)
    Uk = np.array([1,np.exp(+1j*qx/2),np.exp(+1j*qy/2)])
    
    Nk = klin.shape[0]**2
    ret = np.einsum('xybal,xyabm,xylm,a,b->ab',uk,ukq,FFk,Uk,np.conj(Uk),optimize=True) / Nk
    #
    return ret

def filling(energies,vectors,T):
    fk = fermi(energies,T)
    Nk = energies.shape[0]
    mat = np.zeros((3,3,3),dtype=float)
    mat[0,:,:] = np.diag([1,0,0])
    mat[1,:,:] = np.diag([0,1,0]) 
    mat[2,:,:] = np.diag([0,0,1])
    return np.einsum('xyl->',fk)/Nk**2, np.einsum('xyl,xyil,xyjl,aij->a',fk,np.conj(vectors),vectors,mat,optimize=True).real/Nk**2

def filling_fast(energies,T):
    fk = fermi(energies,T)
    Nk = energies.shape[0]
    return np.einsum('xyl->',fk)/Nk**2
    
###################################################
def int_mat(U,J,qx_v,qy_v):
    Jmat = np.zeros((qx_v.shape)+(3,3,),dtype=complex)
    cqx = 2*np.cos(qx_v/2)
    cqy = 2*np.cos(qy_v/2)
    Jmat[:,:,0,1] = cqx
    Jmat[:,:,1,0] = cqx
    Jmat[:,:,0,2] = cqy
    Jmat[:,:,2,0] = cqy
    Umat = np.identity(3)
    Umat = np.einsum('ij,ab->ijab',np.ones(qx_v.shape),Umat)
    return U*Umat + J*Jmat

###################################################
def self_cons_mu(n_wanted,pars,U,J=0,Nk=100,T=0,mu_ini=0,mix=0,Nitermax = 1000,tol = 1e-4):
    
    mu = mu_ini + 0
    LIM = max(4,2*abs(U),4*abs(J))
    
    Ek, phik, _ = make_H(Nk,pars=pars,mud=mu-U*n_wanted/2,mup=mu-U*n_wanted/4)
    ntot, npd = filling(Ek,phik,T=T)
    
    from scipy.optimize import fsolve
    
    for i in range(Nitermax):
        dmud = - U*npd[0] +2*J*npd[1]
        dmup = - U*npd[1] +J*npd[0]
    
        def fn(x):
            Ek, phik, _ = make_H(Nk,pars=pars,mud=x+dmud,mup=x+dmup)
            return filling_fast(Ek,T=T) - n_wanted
        
        mu_new = find_root(fn,[-LIM,LIM])
    
        Ek, phik, _ = make_H(Nk,pars=pars,mud=mu_new+dmud,mup=mu_new+dmup)
        ntot, npd = filling(Ek,phik,T=T)
        print('%d   %.6f    %.6f    %.6f | densities:  %.6f  %.6f | dmus:  %.6f  %.6f' 
              %(i,mu,mu_new,ntot,npd[0],npd[1],dmud,dmup))
    
        if (abs(mu-mu_new)<tol or i==Nitermax-1):
            print('Solution found')
            
            Ek, phik, _ = make_H(Nk,pars=pars,mud=mu_new+dmud,mup=mu_new+dmup)
            ntot, npd = filling(Ek,phik,T=T)
            dmud = - U*npd[0] +2*J*npd[1]
            dmup = - U*npd[1] +J*npd[0]
            def fn(x):
                Ek, phik, _ = make_H(Nk,pars=pars,mud=x+dmud,mup=x+dmup)
                return filling_fast(Ek,T=T) - n_wanted
        
            mu = find_root(fn,[-LIM,LIM])
            mud,mup = mu-U*npd[0]+2*J*npd[1],mu-U*npd[1]+J*npd[0]
            
            print('mud=%.6f,mup=%.6f' %(mud,mup))
            Ek, phik, _ = make_H(Nk,pars=pars,mud=mud,mup=mup)
            ntot, npd = filling(Ek,phik,T=T)
            print(ntot,npd)
            break
    
        else:
            mu = (1-mix)*mu_new + mix*mu

    return mud,mup

###########################################
def filt_inv(x,tol=1e-10):
    return x/(x**2+tol**2)
    
def invert_J(U,J,qsx,qsy):
    eigvcsJ = np.zeros(qsx.shape+(3,3,))
    cx = np.cos(qsx/2)
    cy = np.cos(qsy/2)
    Dk = np.sqrt(cx**2+cy**2)
    cosk = -cx / Dk
    sink = -cy / Dk
    osq2 = 1/np.sqrt(2)
    one = np.ones(qsx.shape)
    zero = np.zeros(qsx.shape)
    eigvcsJ[:,:,:,0] = np.transpose(np.array([one,cosk,sink]) * osq2,(1,2,0))
    eigvcsJ[:,:,:,1] = np.transpose(np.array([one,-cosk,-sink]) * osq2,(1,2,0))
    eigvcsJ[:,:,:,2] = np.transpose(np.array([zero,-sink,cosk]),(1,2,0))
    
    eigvlsJ = np.zeros(qsx.shape+(3,))
    eigvlsJ[:,:,0] = +2*J*Dk + U
    eigvlsJ[:,:,1] = -2*J*Dk + U
    eigvlsJ[:,:,2] = zero + U

    intsr = filt_inv(np.einsum('xyl,lm->xylm',eigvlsJ,np.identity(3)))
    ints = np.einsum('xyil,xylm,xyjm->xyij',eigvcsJ,intsr,eigvcsJ)

    return eigvlsJ,eigvcsJ,ints

###########################################
def gen_path(k_lin):
    mask1 = k_lin>=0
    ids = np.arange(0,k_lin.shape[0])
    mask2 = ids%2 < 100
    mask = np.logical_and(mask1,mask2)
    q_lin = k_lin[mask]
    qxs = np.concatenate((q_lin,0*q_lin[1:]+np.pi,np.pi-q_lin[1:-1]))
    qys = np.concatenate((0*q_lin,q_lin[1:],np.pi-q_lin[1:-1]))

    return qxs,qys

###########################################
def Ppvalue(x): 
    if (x==0):
        ret = 0
    else:
        ret = 1/x
    return ret
            
np.pvalue = np.vectorize(Ppvalue)

def Pvalue(x):
    sh = x.shape
    X = x.flatten()
    mask = X==0
    ret = np.zeros(X.shape)
    ret[mask] = 0
    ret[np.logical_not(mask)] = 1/X[np.logical_not(mask)]
    return np.reshape(ret,sh)