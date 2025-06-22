import numpy as np 

import sys
import myfuncs as myf

class HF_qQ:

    def __init__(self,Nk,pars):
        self.pars = pars
        self.Nk = Nk
        self.init_bands()
        self.Pauli = self.init_Pauli()

    def init_Pauli(self):
        Pauli = np.zeros((4,2,2),dtype=complex)
        Pauli[0,...] = np.identity(2)
        Pauli[1,...] = np.array([[0,1],[1,0]])
        Pauli[2,...] = np.array([[0,-1j],[1j,0]])
        Pauli[3,...] = np.array([[1,0],[0,-1]])
        return Pauli

    def init_bands(self):
        klin = np.linspace(-1,1,self.Nk+1)[1:]*np.pi
        ky,kx = np.meshgrid(klin,klin)
        self.klin = klin
        self.kx = kx
        self.ky = ky
        
        self.H0k = np.zeros((self.Nk,self.Nk,3,3),dtype=complex)
        
        #self.H0k[:,:,0,1] = -self.pars['t']*(1+np.exp(+1j*kx))
        #self.H0k[:,:,1,0] = -self.pars['t']*(1+np.exp(-1j*kx))
        #self.H0k[:,:,0,2] = -self.pars['t']*(1+np.exp(+1j*ky))
        #self.H0k[:,:,2,0] = -self.pars['t']*(1+np.exp(-1j*ky))

        self.H0k[:,:,0,1] = -self.pars['t']*2*np.cos(kx/2)
        self.H0k[:,:,1,0] = -self.pars['t']*2*np.cos(kx/2)
        self.H0k[:,:,0,2] = -self.pars['t']*2*np.cos(ky/2)
        self.H0k[:,:,2,0] = -self.pars['t']*2*np.cos(ky/2)

    def make_H0Q(self,Q):
        kx = self.kx + Q[0]
        ky = self.ky + Q[1]
        
        H0kQ = np.zeros((self.Nk,self.Nk,3,3),dtype=complex)
        
        # H0kQ[:,:,0,1] = -self.pars['t']*(1+np.exp(+1j*kx))
        # H0kQ[:,:,1,0] = -self.pars['t']*(1+np.exp(-1j*kx))
        # H0kQ[:,:,0,2] = -self.pars['t']*(1+np.exp(+1j*ky))
        # H0kQ[:,:,2,0] = -self.pars['t']*(1+np.exp(-1j*ky))
        
        H0kQ[:,:,0,1] = -self.pars['t']*2*np.cos(kx/2)
        H0kQ[:,:,1,0] = -self.pars['t']*2*np.cos(kx/2)
        H0kQ[:,:,0,2] = -self.pars['t']*2*np.cos(ky/2)
        H0kQ[:,:,2,0] = -self.pars['t']*2*np.cos(ky/2)
        
        return H0kQ


    def make_H(self,sc_pars,Q):

        #print(sc_pars)
        H0kQ = self.make_H0Q(Q)
        
        Dd = sc_pars['Dd']
        Dpx = sc_pars['Dpx']
        Dpy = sc_pars['Dpy']
            
        mud = sc_pars['mud']
        mupx = sc_pars['mupx']
        mupy = sc_pars['mupy']
        
        SMat = np.diag([Dd,Dpx,Dpy])
        MuMat = np.diag([mud,mupx,mupy])
        ones = np.ones((self.Nk,self.Nk))
        SMat = np.einsum('xy,ij->xyij',ones,SMat)
        MuMat = np.einsum('xy,ij->xyij',ones,MuMat)
        
        Hret = np.zeros((self.Nk,self.Nk,6,6),dtype=complex)
        Hret[:,:,:3,:3] = self.H0k - MuMat
        Hret[:,:,3:,3:] = H0kQ - MuMat
        Hret[:,:,:3,3:] = -SMat
        Hret[:,:,3:,:3] = -np.conj(SMat)
        
        Es, phis = np.linalg.eigh(Hret)

        return Es,phis,Hret

    def make_H_Q2(self,sc_pars,Q):

        #print(sc_pars)
        H01 = self.make_H0Q(-Q/2)
        H02 = self.make_H0Q(+Q/2)
        
        Dd = sc_pars['Dd']
        Dpx = sc_pars['Dpx']
        Dpy = sc_pars['Dpy']
            
        mud = sc_pars['mud']
        mupx = sc_pars['mupx']
        mupy = sc_pars['mupy']
        
        SMat = np.diag([Dd,Dpx,Dpy])
        MuMat = np.diag([mud,mupx,mupy])
        ones = np.ones((self.Nk,self.Nk))
        SMat = np.einsum('xy,ij->xyij',ones,SMat)
        MuMat = np.einsum('xy,ij->xyij',ones,MuMat)
        
        Hret = np.zeros((self.Nk,self.Nk,6,6),dtype=complex)
        Hret[:,:,:3,:3] = H01 - MuMat
        Hret[:,:,3:,3:] = H02 - MuMat
        Hret[:,:,:3,3:] = -SMat
        Hret[:,:,3:,:3] = -np.conj(SMat)
        
        Es, phis = np.linalg.eigh(Hret)

        return Es,phis,Hret

    def fermi(self,x,T):
        if (T==0):
            t = np.sign(x)
        else:
            t = np.tanh(x/(2*T))
        return 0.5*(1-t)

    def filling(self,Es,phis,T):
        fk = self.fermi(Es,T)
        Nk = self.Nk
        mat = np.zeros((3,6,6),dtype=float)
        mat[0,:,:] = np.diag([1,0,0,1,0,0])
        mat[1,:,:] = np.diag([0,1,0,0,1,0]) 
        mat[2,:,:] = np.diag([0,0,1,0,0,1])
        
        return np.einsum('xyl->',fk)/Nk**2, np.einsum('xyl,xyil,xyjl,aij->a',fk,np.conj(phis),phis,mat,optimize=True).real/Nk**2

    def magnetiz(self,Es,phis,T):
        fk = self.fermi(Es,T)
        Nk = self.Nk
        mat = np.zeros((3,6,6),dtype=complex)
        
        mat[0,3,0] = 1
        mat[1,4,1] = 1
        mat[2,5,2] = 1

        return np.einsum('xyl,xyil,xyjl,aij->a',fk,np.conj(phis),phis,mat,optimize=True)/Nk**2

    def Q_eq(self,Q,sc_pars,T):
        Ek,phik,_ = self.make_H(sc_pars=sc_pars,Q=Q)
        fk = self.fermi(Ek,T)
        Nk = self.Nk
        matQ = np.zeros(self.kx.shape+(2,6,6,),dtype=complex)
        
        # matQ[...,0,3,4] = -1j*self.pars['t']*np.exp(+1j*(self.kx+Q[0]))
        # matQ[...,0,4,3] = +1j*self.pars['t']*np.exp(-1j*(self.kx+Q[0]))
        # matQ[...,1,3,5] = -1j*self.pars['t']*np.exp(+1j*(self.ky+Q[1]))
        # matQ[...,1,5,3] = +1j*self.pars['t']*np.exp(-1j*(self.ky+Q[1]))

        matQ[...,0,3,4] = -self.pars['t']*np.sin( (self.kx+Q[0])/2 )
        matQ[...,0,4,3] = -self.pars['t']*np.sin( (self.kx+Q[0])/2 )
        matQ[...,1,3,5] = -self.pars['t']*np.sin( (self.ky+Q[1])/2 )
        matQ[...,1,5,3] = -self.pars['t']*np.sin( (self.ky+Q[1])/2 )

        return np.einsum('xyl,xyil,xyjl,xyaij->a',fk,np.conj(phik),phik,matQ,optimize=True)/Nk**2
        

    def filling_fast(self,energies,T):
        fk = self.fermi(energies,T)
        Nk = energies.shape[0]
        return np.einsum('xyl->',fk)/Nk**2

    
    def Free_Energy(self,Ek,T):
        fk = self.fermi(Ek,T)
        if (abs(T)<1e-2):
            return 1/self.Nk**2 * np.einsum('xyl,xyl->',Ek,fk)
        else:
            return -T/self.Nk**2 * np.einsum('xyl->',np.log(1+np.exp(-Ek/T)))

    def random_complex(self,nmax):
        re = (2*np.random.rand()-1)*nmax
        im = (2*np.random.rand()-1)*nmax

        return re + 1j *im
            
    def random_init_cond(self):
        Ddi = self.random_complex(1)
        Dpxi = self.random_complex(1)
        Dpyi = self.random_complex(1)
        
        return dict(mu=0,Dd=Ddi,Dpx=Dpxi,Dpy=Dpyi)
        
    ##############################################################################################
    def SC_loop_findQ(self,n_wanted,T,par_ini=None,Qx=None,Qy=None,Qdiag=False,mix=0,Nitermax = 1000,tol = 1e-4,
                      vb=False,Hartree_term=True,symm=False,symm2=False,vb_findroot=False,LIMQ=None):
        if par_ini is None:
            print('random initial conditions')
            par_ini = self.random_init_cond()
            
        mu = par_ini['mu'] + 0
        U = self.pars['U']
        J = self.pars['J']
        J_pxpy = self.pars['J_pxpy']
        #mas chem pot in the search
        LIM = max(4,8*abs(U),8*abs(J),100)
        #initialize chemical potentials, gaps, and Hamiltonian
        sc_pars=dict(mud=0,mupx=0,mupy=0,Dd=0,Dpx=0,Dpy=0)
        
        sc_pars['mud']=mu - U*n_wanted/2
        sc_pars['mupx']=mu - U*n_wanted/4
        sc_pars['mupy']=mu - U*n_wanted/4
        
        if 'mud' in par_ini:
            sc_pars['mud'] = par_ini['mud'] 
        if 'mupx' in par_ini:
            sc_pars['mupx'] = par_ini['mupx'] 
        if 'mupy' in par_ini:
            sc_pars['mupy'] = par_ini['mupy']

        dmud = sc_pars['mud'] - mu
        dmupx = sc_pars['mupx'] - mu
        dmupy = sc_pars['mupy'] - mu
        
        Dd = par_ini['Dd']
        Dpx = par_ini['Dpx']
        Dpy = par_ini['Dpy']
        sc_pars['Dd'] = Dd
        sc_pars['Dpx'] = Dpx
        sc_pars['Dpy'] = Dpy

        HT = float(Hartree_term)
        #print(HT)

        findQx=False
        findQy=False
        
        if (Qx is None):
            Qx=(2*np.random.rand()-1)*np.pi
            findQx=True
        if (Qy is None):
            Qy=(2*np.random.rand()-1)*np.pi
            findQy=True

        if (Qdiag):
            findQx=False
            findQy=False
            Qd=(2*np.random.rand()-1)*np.pi
            Qx=Qd
            Qy=Qd
            
        Q = np.array([Qx,Qy])
        
        #print(sc_pars)
        Ek, phik, _ = self.make_H(sc_pars,Q)
        ntot, npd = self.filling(Ek,phik,T=T)
        
        
        for i in range(Nitermax):

            
            def fn(x):
                sc_pars['mud'] = dmud*HT
                sc_pars['mupx'] = dmupx*HT
                sc_pars['mupy'] = dmupy*HT
                sc_pars['Dd'] = Dd
                sc_pars['Dpx'] = Dpx
                sc_pars['Dpy'] = Dpy
                Ek, phik, _ = self.make_H(sc_pars,Q)
                return self.filling_fast(Ek-x,T=T) - 3*n_wanted

            
            mu = myf.find_root(fn,[-LIM,LIM],tol=1e-5,vb=False)

            def fQx(qx):
                sc_pars['mud'] = dmud*HT + mu
                sc_pars['mupx'] = dmupx*HT + mu
                sc_pars['mupy'] = dmupy*HT + mu
                sc_pars['Dd'] = Dd
                sc_pars['Dpx'] = Dpx
                sc_pars['Dpy'] = Dpy
                ret = self.Q_eq([qx,Q[1]],sc_pars,T)[0].real
                return ret

            if (LIMQ is None):
                LIMQ1=-0.999*np.pi
                LIMQ2=+0.9999*np.pi
            else:
                LIMQ1=LIMQ[0]
                LIMQ2=LIMQ[1]
            #qsearch=self.klin[self.klin>0]
            qsearch=np.linspace(0.005,1,100)*np.pi
            if (findQx):
                Qx = myf.find_root(fQx,[LIMQ1,LIMQ2],tol=1e-5,vb=vb_findroot)
                Q[0] = Qx
                #derivative=np.zeros(qsearch.shape[0])
                #for cnt,qq in enumerate(qsearch):
                #    derivative[cnt] = fQx(qq)
                #Q[0] = qsearch[np.argmin(abs(derivative))]
                
            #print('Q=',Q)

            def fQy(qy):
                sc_pars['mud'] = dmud*HT + mu
                sc_pars['mupx'] = dmupx*HT + mu
                sc_pars['mupy'] = dmupy*HT + mu
                sc_pars['Dd'] = Dd
                sc_pars['Dpx'] = Dpx
                sc_pars['Dpy'] = Dpy
                return self.Q_eq([Q[0],qy],sc_pars,T)[1].real

            if (findQy):
                Qy = myf.find_root(fQy,[LIMQ1,LIMQ2],tol=1e-5,vb=vb_findroot)
                Q[1] = Qy
                #derivative=np.zeros(qsearch.shape[0])
                #for cnt,qq in enumerate(qsearch):
                #    derivative[cnt] = fQy(qq)
                #Q[1] = qsearch[np.argmin(abs(derivative))]


            def fQd(qd):
                sc_pars['mud'] = dmud*HT + mu
                sc_pars['mupx'] = dmupx*HT + mu
                sc_pars['mupy'] = dmupy*HT + mu
                sc_pars['Dd'] = Dd
                sc_pars['Dpx'] = Dpx
                sc_pars['Dpy'] = Dpy
                return np.sum(self.Q_eq([qd,qd],sc_pars,T).real)/np.sqrt(2)

            if (Qdiag):
                Qd = myf.find_root(fQd,[LIMQ1,LIMQ2],tol=1e-5,vb=vb_findroot)
                Q[0] = Qd
                Q[1] = Qd
                # derivative=np.zeros(qsearch.shape[0])
                # for cnt,qq in enumerate(qsearch):
                #     derivative[cnt] = fQd(qq)
                # Q[0] = qsearch[np.argmin(abs(derivative))]
                # Q[1] = qsearch[np.argmin(abs(derivative))]

            sc_pars['mud'] = mu+dmud*HT
            sc_pars['mupx'] = mu+dmupx*HT
            sc_pars['mupy'] = mu+dmupy*HT
            sc_pars['Dd'] = Dd
            sc_pars['Dpx'] = Dpx
            sc_pars['Dpy'] = Dpy
            Ek, phik, _ = self.make_H(sc_pars,Q)
            ntot, npd = self.filling(Ek,phik,T=T)
            Mags = self.magnetiz(Ek,phik,T=T)
            F = self.Free_Energy(Ek,T=T) + mu * ntot + (U) * (np.sum(abs(Mags)**2) - np.sum(abs(npd/2)**2))

            #update the FM OPs!!!
            Dd_new  = U*Mags[0]-2*J*(Mags[1]+Mags[2])
            Dpx_new = U*Mags[1]-2*J*Mags[0] - 4*J_pxpy*Mags[2]
            Dpy_new = U*Mags[2]-2*J*Mags[0] - 4*J_pxpy*Mags[1]

            #update the mu shifts
            dmud_new = - U*npd[0]/2*HT# +J*(npd[1]+npd[2])
            dmupx_new = - U*npd[1]/2*HT# +J*npd[0]
            dmupy_new = - U*npd[2]/2*HT# +J*npd[0]

            if (symm):
                dmupy_new = dmupx_new
                Dpy_new = Dpx_new

            if (symm2):
                dmupy_new = dmupx_new
                phase = np.exp(+2*1j*np.angle(Dd_new))
                Dpy_new = np.conj(Dpx_new) * phase
                
            ctrl = max(abs(dmud_new-dmud),abs(dmupx_new-dmupx),abs(dmupy_new-dmupy),
                       abs(Dd_new-Dd),abs(Dpx_new-Dpx),abs(Dpy_new-Dpy))

            # if (vb):
            #     print('\t D_dmud: %.5f\n\t D_dmupx: %.5f\n\t D_dmupy: %.5f\n\t  D_Dd: %.5f\n\t D_Dpx: %.5f\n\t D_Dpy: %.5f' %(abs(dmud_new-dmud),abs(dmupx_new-dmupx),abs(dmupy_new-dmupy),abs(Dd_new-Dd),abs(Dpx_new-Dpx),abs(Dpy_new-Dpy)) )
            #     print('\t\t dmupx=%.5f, dmupx_old=%.5f, dmupy=%.5f, dmupy_old=%.5f' %(dmud_new,dmupx,dmupy_new,dmupy))
            #     print('\t\t |Dpx-Dpx_old| = %.5f, |Dpx-Dpx_old| = %.5f' %(abs(Dpx_new-Dpx),abs(Dpy_new-Dpy)))
            
            if (vb):
                #print('\n',abs(dmud_new-dmud),abs(dmupx_new-dmupx),abs(dmupy_new-dmupy),
                #       abs(Dd_new-Dd),abs(Dpx_new-Dpx),abs(Dpy_new-Dpy),'\n')
                phase = np.exp(-1j*np.angle(Dd_new))
                #print('Q=',Q)
                print('%d %.5f : %.4f  %.4f|%.4f|[%.3f,%.3f]| dmus %.4f  %.4f %.4f |(%.4f, %.4f) (%.4f, %.4f) (%.4f, %.4f)' 
                      %(i,ctrl,mu,ntot/3,F,Q[0]/np.pi,Q[1]/np.pi,dmud_new,dmupx_new,dmupy_new,abs(Dd_new),np.angle(Dd_new),
                        abs(Dpx_new),np.angle(Dpx_new*phase),abs(Dpy_new),np.angle(Dpy_new*phase)))

            
            if (ctrl<tol or i==Nitermax-1):
                if (vb and ctrl<tol):
                    print('Solution found')
                if (i==Nitermax-1):
                    #raise ValueError('SC loop: Not converged')
                    print('SC loop: Not converged',pars)

                dmud,dmupx,dmupy = dmud_new,dmupx_new,dmupy_new
                mud,mupx,mupy = mu+dmud*HT,mu+dmupx*HT,mu+dmupy*HT
                Dd, Dpx, Dpy = Dd_new,Dpx_new,Dpy_new
                if (vb):
                     print('mu = %.6f' %mu)
                     print('mud=%.6f, mupx=%.6f, mupy=%.6f' %(mud,mupx,mupy))
                     print('Dd=(%.6f, %.6f), Dpx=(%.6f, %.6f), Dpy=(%.6f, %.6f)' %(abs(Dd_new),np.angle(Dd_new),abs(Dpx_new),
                                                                np.angle(Dpx_new),abs(Dpy_new),np.angle(Dpy_new)))
                sc_pars['mud'] = mud
                sc_pars['mupx'] = mupx
                sc_pars['mupy'] = mupy
                sc_pars['Dd'] = Dd
                sc_pars['Dpx'] = Dpx
                sc_pars['Dpy'] = Dpy
                Ek, phik, _ = self.make_H(sc_pars,Q)
                ntot, npd = self.filling(Ek,phik,T=T)
                Mags = self.magnetiz(Ek,phik,T=T)
                F = self.Free_Energy(Ek,T=T) + mu * ntot + (U) * (np.sum(abs(Mags)**2) - np.sum(abs(npd/2)**2)) 
                if (vb):
                    print('free energy = ', F)
                    print('densities: ',ntot/3,npd)
                    phase=np.exp(-1j*np.angle(Mags[0]))
                    print('Magnetizations: ',np.sqrt(Mags*np.conj(Mags)).real,np.angle(phase*Mags))
                break
        
            else:
                dmud = (1-mix)*dmud_new + mix*dmud
                dmupx = (1-mix)*dmupx_new + mix*dmupx
                dmupy = (1-mix)*dmupy_new + mix*dmupy
                
                Dd = (1-mix)*Dd_new + mix*Dd
                Dpx = (1-mix)*Dpx_new + mix*Dpx
                Dpy = (1-mix)*Dpy_new + mix*Dpy
                # Dd=Dd_new
                # Dpx=Dpx_new
                # Dpy=Dpy_new

                if (symm):
                    dmupy = dmupx
                    Dpy = Dpx
                if (symm2):
                    dmupy = dmupx
                    phase = np.exp(1j*2*np.angle(Dd))
                    Dpy = np.conj(Dpx)*phase

        return mud,mupx,mupy,Dd,Dpx,Dpy,F,mu,Q[0],Q[1]
        
    def calc_chi_SL(self,Ek,phik,T):
        #Ek, phik, _ = self.make_H(SC_pars,Q)
        phik_resh = np.zeros(phik.shape[:2] + (3,2,) + phik.shape[3:],dtype=complex)
        phik_resh[:,:,:,0,:] = phik[:,:,:3,:]
        phik_resh[:,:,:,1,:] = phik[:,:,3:,:]
        fEk = self.fermi(Ek,T)
        Ak = np.einsum('xyisl,xyl,xyjpl->xyijsp',np.conj(phik_resh),fEk,phik_resh,optimize=True)
        Norm = self.Nk**2
        chi_ii = np.einsum('xyiisp->isp',Ak,optimize=True)/Norm
        Mat_od = np.zeros((8,3,3))
        Mat_od[0,0,1] = 1
        Mat_od[1,1,0] = 1
        Mat_od[2,0,2] = 1
        Mat_od[3,2,0] = 1
        #
        Mat_od[4,0,1] = 1
        Mat_od[5,1,0] = 1
        Mat_od[6,0,2] = 1
        Mat_od[7,2,0] = 1
        #
        ffk = np.zeros((8,)+Ek.shape[:2],dtype=complex)
        ffk[0,...] = np.exp(+1j*self.kx/2)
        ffk[1,...] = np.exp(-1j*self.kx/2)
        ffk[2,...] = np.exp(+1j*self.ky/2)
        ffk[3,...] = np.exp(-1j*self.ky/2)
        #
        ffk[4,...] = np.exp(-1j*self.kx/2)
        ffk[5,...] = np.exp(+1j*self.kx/2)
        ffk[6,...] = np.exp(-1j*self.ky/2)
        ffk[7,...] = np.exp(+1j*self.ky/2)
        #
        chi_ij = np.einsum('xyijsp,aji,axy->asp',Ak,Mat_od,ffk,optimize=True)/Norm

        #rotate order onto xz plane
        #U = np.array([[1-1j,-1-1j],[1-1j,1+1j]]) / 2
        U = np.identity(2)

        chi_ii = np.einsum('ab,ibc,cd->iad',np.conj(U.T),chi_ii,U)
        chi_ij = np.einsum('ab,ibc,cd->iad',np.conj(U.T),chi_ij,U)

        sym_time_der = (chi_ii[:,0,0]-chi_ii[:,1,1])
        nonsym_time_der = chi_ii[:,0,1]

        #sym_hopping = chi_ij[:,0,0]+chi_ij[[1,0,3,2],1,1]
        #nonsym_hopping = chi_ij[:,0,1]-chi_ij[[1,0,3,2],0,1]
        sym_hopping = chi_ij[:,0,0]+chi_ij[[1,0,3,2,5,4,7,6],1,1]
        nonsym_hopping = chi_ij[:,0,1]-chi_ij[[1,0,3,2,5,4,7,6],0,1]
        
        return chi_ii, chi_ij, sym_time_der,nonsym_time_der,sym_hopping,nonsym_hopping

    def gen_Q_path(self):
        klin2=self.klin[self.Nk//2-1:]
        Qx = np.concatenate((klin2,klin2[1:]*0+np.pi,klin2[::-1][1:-1]))
        Qy = np.concatenate((klin2*0,klin2[1:],klin2[::-1][1:-1]))

        return Qx,Qy