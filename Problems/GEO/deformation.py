import numpy as np


def calc_deformation(time: list, head: list, Kv: float, Sskv: float,
                     Sske: float, claythick: int, nclay: int,
                     sandthick: int = -1, Nt_const: int = 70):
    """
    Calculate the deformation of clay layer of user-defined thickness.

    Args:
        time (list): A vector of the times of the head measurments.
        head (list): A vector (same length as time) with head measurements.
        Kv (float): The vertical hydraulic conductivity.
        Sskv (float): The skeletal specific storage (inelastic).
        Sske (float): The skeletal specific storage (elastic).
        clay_thick (int): The thickness of a single clay layer modeled.
        nclay (int): The number of clay layers.
        sandthick (int): The total thickness of sand. If "< 0", then
            it's equal to clay thickness.
        t (list): A vector containing the time at each time step.
            Needs to be sampled very dense.
            A dt of around 0.1 days usually is sufficient.
        Nz (int): number of layers in z direction, within the single clay
            layer modeled. Higher Nz requires higher dt, and vice versa

    Returns:
        (list, list, list, list): The interpolated time, total surface
            deformation, interpolated head, and inelastic surface deformation.
    """
    Sske_sand_mult = 0.5  # assuem sand sske
    Sske_sand = Sske*Sske_sand_mult
    if sandthick < 0:
        sand_thick = nclay*claythick
    else:
        sand_thick = sandthick
    Ske_sand = Sske_sand*sand_thick

    D = Kv/Sske
    num_yrs = time[-1]-time[0]
    Nz = 5  # discretization of clay layer.
    # Higher discretization in z requires higher discretization in time
    # (longer simulation)

    clay_thick = claythick/2  # simulate doubly draining clay
    z = np.linspace(0, clay_thick, Nz+2)    # mesh points in space
    dz = clay_thick/Nz
    dz_full = np.ones(np.shape(z)) * dz
    dz_full[0] = dz/2
    dz_full[-1] = dz/2
    tau = np.square(claythick/2)*Sske/Kv

    if tau < .1:  # assume instantaneous equilibration of clay if tau <.1
        D = Kv/Sskv
        # set optimum number of time steps based on diffusivity.
        # Minimum number is 4 per year
        Nt = int(num_yrs*24+D*Nt_const)
        t = np.linspace(time[0], time[-1], int(Nt+1))

        boundary = np.interp(t, time, head)

        dt = t[1] - t[0]
        h = boundary[0]*np.ones((Nz+2,Nt+1))  
        h[0, :] = boundary

        precons_head = h[:, 0].copy()

        deformation = np.zeros(np.shape(h))
        deformation_v = np.zeros(np.shape(h))

        for n in range(0, Nt):
            # Compute u at inner mesh points
            for i in range(1, Nz+1):
                dz1=dz/(int(i==1)+1)
                dz2=dz
                dz_all=np.mean([dz1,dz2])
                h_new = h[0,n+1]
                dh=(h_new-h[i,n])
                defm=dh*Sske*dz*2
                deformation_v[i,n+1]=deformation_v[i,n]
                if np.logical_and(h_new<precons_head[i],dh<0): # if head drops below preconsolidation head, then Ss changes to Sskv
                    h_new = h[i,n] + ((Kv/Sskv)*dt/dz_all)*((h[i-1,n] - h[i,n])/dz1+( - h[i,n] + h[i+1,n])/dz2)
                    precons_head[i]=h_new
                    dh=(h_new-h[i,n])
                    defm=dh*Sskv*dz*2
                    deformation_v[i,n+1]=defm+np.min(deformation_v[i,0:(n+1)])
                h[i,n+1]=h_new
                deformation[i,n+1]=defm+deformation[i,n]
            h[-1,n+1]=h_new
    else:    
        Nt=int(num_yrs*24+D*Nt_const) # set optimum number of time steps based on diffusivity. Minimum number is 4 per year

        t=np.linspace(time[0],time[-1],int(Nt+1))

        boundary=np.interp(t,time,head)

        dt = t[1] - t[0]
        h = boundary[0]*np.ones((Nz+2,Nt+1))  
        h[0,:]=boundary
        
        precons_head=h[:,0].copy()
    
        deformation=np.zeros(np.shape(h))
        deformation_v=np.zeros(np.shape(h))
        
        for n in range(0, Nt):
            # Compute u at inner mesh points
            for i in range(1, Nz+1):
                dz1=dz/(int(i==1)+1)
                dz2=dz
                dz_all=np.mean([dz1,dz2])
                h_new = h[i,n] + ((Kv/Sske)*dt/dz_all)*((h[i-1,n] - h[i,n])/dz1+( - h[i,n] + h[i+1,n])/dz2)
                dh=(h_new-h[i,n])
                defm=dh*Sske*dz*2
                deformation_v[i,n+1]=deformation_v[i,n]
                if np.logical_and(h_new<precons_head[i],dh<0): # if head drops below preconsolidation head, then Ss changes to Sskv
                    h_new = h[i,n] + ((Kv/Sskv)*dt/dz_all)*((h[i-1,n] - h[i,n])/dz1+( - h[i,n] + h[i+1,n])/dz2)
                    precons_head[i]=h_new
                    dh=(h_new-h[i,n])
                    defm=dh*Sskv*dz*2
                    deformation_v[i,n+1]=defm+np.min(deformation_v[i,0:(n+1)])
                h[i,n+1]=h_new
                deformation[i,n+1]=defm+deformation[i,n]
            h[-1,n+1]=h_new
    deformation=np.sum(deformation,axis=0)*nclay
    deformation_v=np.sum(deformation_v,axis=0)*nclay
    boundary0=boundary-boundary[0]
    deformation=deformation+boundary0*Ske_sand
    return(t,deformation,boundary,deformation_v)