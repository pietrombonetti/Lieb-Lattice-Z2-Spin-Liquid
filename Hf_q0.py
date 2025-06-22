import numpy as np 
import matplotlib.pyplot as plt

import sys
import myfuncs as myf

class HF_q0:

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
        self.cx = np.cos(kx/2)
        self.cy = np.cos(ky/2)
        self.CX = np.cos(kx)
        self.CY = np.cos(ky)
        self.klin = klin
        self.kx = kx
        self.ky = ky
        
        self.H0k = np.zeros((self.Nk,self.Nk,3,3),dtype=complex)
        
        self.H0k[:,:,0,1] = -2*self.pars['t']*self.cx
        self.H0k[:,:,1,0] = -2*self.pars['t']*self.cx
        self.H0k[:,:,0,2] = -2*self.pars['t']*self.cy
        self.H0k[:,:,2,0] = -2*self.pars['t']*self.cy

        self.H0k[:,:,0,0] = -2*self.pars['t_dd']*(self.CX+self.CY)
        self.H0k[:,:,1,1] = -2*self.pars['t_pp']*self.CX
        self.H0k[:,:,2,2] = -2*self.pars['t_pp']*self.CY
        self.H0k[:,:,1,2] = -4*self.pars['t_pxpy']*self.cx*self.cy
        self.H0k[:,:,2,1] = -4*self.pars['t_pxpy']*self.cx*self.cy

    def make_H(self,sc_pars):
        
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
        Hret[:,:,3:,3:] = self.H0k - MuMat
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

        # with this (wrong) choice I get an altermagnet!!!
        #mat[0,0,3] = 1
        #mat[1,1,4] = 1
        #mat[2,2,5] = 1
        
        return np.einsum('xyl,xyil,xyjl,aij->a',fk,np.conj(phis),phis,mat,optimize=True)/Nk**2
        

    def filling_fast(self,energies,T):
        fk = self.fermi(energies,T)
        Nk = energies.shape[0]
        return np.einsum('xyl->',fk)/Nk**2

    
    def Free_Energy(self,Ek,T=0):
        fk = self.fermi(Ek,T)
        if (T==0):
            return 1/self.Nk**2 * np.einsum('xyl,xyl->',Ek,fk)
        else:
            #expEk=np.exp(-Ek/T)
            #t=np.tanh(Ek/(2*T))
            #expEk=(1-t)*np.pvalue(1+t)
            #logEk= np.log(2) - np.log(1+np.tanh(Ek/(2*T)))
            logEk=-T*np.log(1+np.exp(-Ek/T))
            return 1/self.Nk**2 * np.einsum('xyl->',logEk)

    def random_complex(self,nmax):
        re = (2*np.random.rand()-1)*nmax
        im = (2*np.random.rand()-1)*nmax

        return re + 1j *im
            
    def random_init_cond(self):
        Ddi = self.random_complex(1)
        Dpxi = self.random_complex(1)
        Dpyi = self.random_complex(1)
        
        return dict(mu=0,Delta_d=Ddi,Delta_px=Dpxi,Delta_py=Dpyi)
    
    def SC_loop(self,n_wanted,par_ini=None,T=0,mix=0,Nitermax = 1000,tol = 1e-4,vb=False,Hartree_term=True,symm=False,symm2=False):
        if par_ini is None:
            par_ini = self.random_init_cond()
        
        mu = par_ini['mu'] + 0
        U = self.pars['U']
        J = self.pars['J']
        J_pxpy = self.pars['J_pxpy']
        #mas chem pot in the search
        LIM = max(4,8*abs(U),8*abs(J),100)
        #initialize chemical potentials, gaps, and Hamiltonian
        sc_pars=dict(mud=0,mupx=0,mupy=0,Dp=0,Dpx=0,Dpy=0)
        
        dmud = - U*n_wanted/2
        dmupx = - U*n_wanted/4
        dmupy = - U*n_wanted/4
        sc_pars['mud']=mu + dmud 
        sc_pars['mupx']=mu + dmupx
        sc_pars['mupy']=mu + dmupy
        
        Dd = par_ini['Delta_d']
        Dpy = par_ini['Delta_px']
        Dpx = par_ini['Delta_py']
        sc_pars['Dd'] = Dd
        sc_pars['Dpx'] = Dpx
        sc_pars['Dpy'] = Dpy

        HT = float(Hartree_term)
        #print(HT)

        
        #print(sc_pars)
        Ek, phik, _ = self.make_H(sc_pars)
        ntot, npd = self.filling(Ek,phik,T=T)
        
        
        for i in range(Nitermax):
            
            def fn(x):
                sc_pars['mud'] = dmud*HT
                sc_pars['mupx'] = dmupx*HT
                sc_pars['mupy'] = dmupy*HT
                sc_pars['Dd'] = Dd
                sc_pars['Dpx'] = Dpx
                sc_pars['Dpy'] = Dpy
                Ek, phik, _ = self.make_H(sc_pars)
                return self.filling_fast(Ek-x,T=T) - 3*n_wanted

            #print('=============================')
            #if (T==0):
            #    lin_E = np.sort(Ek.flatten())
            #    Num_p = int(3*n_wanted*self.Nk**2)
            #    mu_new = lin_E[Num_p]-lin_E[0]
            #else:
            mu = myf.find_root(fn,[-LIM,LIM],tol=1e-5)#,vb=True)
            #print('=============================')

            sc_pars['mud'] = mu+dmud*HT
            sc_pars['mupx'] = mu+dmupx*HT
            sc_pars['mupy'] = mu+dmupy*HT
            sc_pars['Dd'] = Dd
            sc_pars['Dpx'] = Dpx
            sc_pars['Dpy'] = Dpy
            Ek, phik, _ = self.make_H(sc_pars)
            ntot, npd = self.filling(Ek,phik,T=T)
            Mags = self.magnetiz(Ek,phik,T=T)
            F = self.Free_Energy(Ek,T=T) + mu * ntot + U * (np.sum(abs(Mags)**2) - np.sum(abs(npd)**2))

            #update the FM OPs!!!
            Dd_new  = U*Mags[0]-2*J*(Mags[1]+Mags[2])
            Dpx_new = U*Mags[1]-2*J*Mags[0] - 4*J_pxpy*Mags[2]
            Dpy_new = U*Mags[2]-2*J*Mags[0] - 4*J_pxpy*Mags[1]

            #update the mu shifts
            dmud_new = - U*npd[0]*HT# +J*(npd[1]+npd[2])
            dmupx_new = - U*npd[1]*HT# +J*npd[0]
            dmupy_new = - U*npd[2]*HT# +J*npd[0]

            if (symm):
                dmupy_new = dmupx_new
                Dpy_new = Dpx_new

            if (symm2):
                dmupy_new = dmupx_new
                phase = np.exp(+2*1j*np.angle(Dd_new))
                Dpy_new = np.conj(Dpx_new) * phase
                
            ctrl = max(abs(dmud_new-dmud),abs(dmupx_new-dmupx),abs(dmupy_new-dmupy),
                       abs(Dd_new-Dd),abs(Dpx_new-Dpx),abs(Dpy_new-Dpy))
            
            if (vb):
                #print('\n',abs(dmud_new-dmud),abs(dmupx_new-dmupx),abs(dmupy_new-dmupy),
                #       abs(Dd_new-Dd),abs(Dpx_new-Dpx),abs(Dpy_new-Dpy),'\n')
                phase = np.exp(-1j*np.angle(Dd_new))
                print('%d %.5f : %.4f  %.4f| %.4f | dmus %.4f  %.4f %.4f | FM OPs:  (%.4f, %.4f) (%.4f, %.4f) (%.4f, %.4f)' 
                      %(i,ctrl,mu,ntot/3,F,dmud_new,dmupx_new,dmupy_new,abs(Dd_new),np.angle(Dd_new),
                        abs(Dpx_new),np.angle(Dpx_new*phase),abs(Dpy_new),np.angle(Dpy_new*phase)))

            
            if (ctrl<tol or i==Nitermax-1):
                if (vb and ctrl<tol):
                    print('Solution found')
                if (i==Nitermax-1):
                    raise ValueError('SC loop: Not converged')

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
                Ek, phik, _ = self.make_H(sc_pars)
                ntot, npd = self.filling(Ek,phik,T=T)
                Mags = self.magnetiz(Ek,phik,T=T)
                F = self.Free_Energy(Ek,T=T) + mu * ntot + U * (np.sum(abs(Mags)**2) - np.sum(abs(npd)**2)) 
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

                if (symm):
                    dmupy = dmupx
                    Dpy = Dpx
                if (symm2):
                    dmupy = dmupx
                    phase = np.exp(1j*2*np.angle(Dd))
                    Dpy = np.conj(Dpx)*phase

        return mud,mupx,mupy,Dd,Dpx,Dpy,F,mu
        
    def calc_chi_SL(self,Ek,phik,T=0):
        #Ek, phik, _ = self.make_H(SC_pars)
        phik_resh = np.zeros(phik.shape[:2] + (3,2,) + phik.shape[3:],dtype=complex)
        phik_resh[:,:,:,0,:] = phik[:,:,:3,:]
        phik_resh[:,:,:,1,:] = phik[:,:,3:,:]
        fEk = self.fermi(Ek,T)
        Ak = np.einsum('xyisl,xyl,xyjpl->xyijsp',np.conj(phik_resh),fEk,phik_resh,optimize=True)
        #print(Ak.shape)
        Norm = self.Nk**2
        chi_ii = np.einsum('xyiisp->isp',Ak,optimize=True)/Norm
        Mat_od = np.zeros(Ek.shape[:2]+(8,3,3,),dtype=complex)
        Mat_od[...,0,0,1] = np.exp(+1j*self.kx/2)
        Mat_od[...,1,1,0] = np.exp(-1j*self.kx/2)
        Mat_od[...,2,0,2] = np.exp(+1j*self.ky/2)
        Mat_od[...,3,2,0] = np.exp(-1j*self.ky/2)
        #
        Mat_od[...,4,0,1] = np.exp(-1j*self.kx/2)
        Mat_od[...,5,1,0] = np.exp(+1j*self.kx/2)
        Mat_od[...,6,0,2] = np.exp(-1j*self.ky/2)
        Mat_od[...,7,2,0] = np.exp(+1j*self.ky/2)
        
        chi_ij = np.einsum('xyijsp,xyaij->asp',Ak,Mat_od,optimize=True)/Norm

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