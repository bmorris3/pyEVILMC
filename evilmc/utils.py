import numpy as np

__all__ = ['orb_pos', 'keplereq', 'cos_f', 'sin_f']

def orb_pos(semi, ecc, asc_node, peri_long, inc, mean_anom):
    # ;+
    # ; NAME:
    # ;	orb_pos
    # ;
    # ; PURPOSE:
    # ;	This routine returns the 3-D position vector of an orbital body, given the orbital
    # ;	elements. Uses equations worked out in Murray & Dermott (1999) _Solar System Dynamics_,
    # ;	ch. 2. Available with no warranties whatsoever at
    # ;	http://www.lpl.arizona.edu/~bjackson/code/idl.html.
    # ;
    # ; CATEGORY:
    # ;	Astrophysics.
    # ;
    # ; CALLING SEQUENCE:
    # ;
    # ;	Result = orb_pos(semi, ecc, asc_node, peri_long, inc, mean_anom)
    # ;
    # ; INPUTS:
    # ;	semi:		orbital semi-major axis
    # ;	ecc:		eccentricity
    # ;	asc_node:	longitude of ascending node, in degrees
    # ;	peri_long: 	longitude of pericenter, in degrees
    # ;	inc: 		orbital inclination, in degrees
    # ;	mean_anom: 	orbital mean anomaly
    # ;
    # ; OUTPUTS:
    # ;	3 x n_elements(mean_anom) array with x, y, z position of orbital body
    # ;
    # ; RESTRICTIONS:
    # ;	Code doesn't do any checking of parameters, so do them yourself. The code requires a
    # ;	companion routine keplereq.pro (written by Marc Buie and Joern Wilms, among others)
    # ;	available here: http://www.lpl.arizona.edu/~bjackson/code/idl.html.
    # ;
    # ; EXAMPLE:
    # ;	semi = 1.0
    # ;	ecc = 0.0 ;orbital eccentricity
    # ;	asc_node = 0. ;longitude of planetary ascending node
    # ;	peri_long = 0. ;longitude of planetary pericenter
    # ;	inc = 83.1 ;orbital inclination in degrees
    # ;	mean_anom = 2.*!pi*dindgen(101)
    # ;	r = orb_pos(semi, ecc, asc_node, peri_long, inc, mean_anom)
    # ;
    # ; MODIFICATION HISTORY:
    # ; 	Written by:	Brian Jackson (decaelus@gmail.com), 2011 July 25.
    # ;-

    #;calculate true anomaly as a function of time, i.e. solve Kepler's equation
    ecc_anom = keplereq(mean_anom, ecc)
    cos_f = (np.cos(ecc_anom)-ecc)/(1.-ecc*np.cos(ecc_anom))
    sin_f = (np.sqrt(1.-ecc**2.)*np.sin(ecc_anom))/(1.-ecc*np.cos(ecc_anom))

    #;cos(inclination)
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)

    #;cos(longitude of ascending node)
    cos_asc = np.cos(asc_node)
    sin_asc = np.sin(asc_node)

    #;longitude of pericenter
    cos_peri = np.cos(peri_long)
    sin_peri = np.sin(peri_long)

    #;cos(peri_long + f)
    cos_wf = cos_peri*cos_f-sin_peri*sin_f
    sin_wf = sin_peri*cos_f+cos_peri*sin_f

    r = np.zeros((3, len(mean_anom)))
    r[0, :] = semi*(1.-ecc**2.)/(1.+ecc*cos_f)*(cos_asc*cos_wf-sin_asc*sin_wf*cos_inc)
    r[1, :] = semi*(1.-ecc**2.)/(1.+ecc*cos_f)*(sin_asc*cos_wf+cos_asc*sin_wf*cos_inc)
    r[2, :] = semi*(1.-ecc**2.)/(1.+ecc*cos_f)*sin_wf*sin_inc

    return r


def keplereq(m,ecc,thresh=1e-5):
    # ;+
    # ; NAME:
    # ;    keplereq
    # ; PURPOSE:
    # ;    Solve Kepler's Equation
    # ; DESCRIPTION:
    # ;    Solve Kepler's Equation. Method by S. Mikkola (1987) Celestial
    # ;       Mechanics, 40 , 329-334.
    # ;    result from Mikkola then used as starting value for
    # ;       Newton-Raphson iteration to extend the applicability of this
    # ;       function to higher eccentricities
    # ;
    # ; CATEGORY:
    # ;    Celestial Mechanics
    # ; CALLING SEQUENCE:
    # ;    eccanom=keplereq(m,ecc)
    # ; INPUTS:
    # ;    m    - Mean anomaly (radians; can be an array)
    # ;    ecc  - Eccentricity
    # ; OPTIONAL INPUT PARAMETERS:
    # ;
    # ; KEYWORD INPUT PARAMETERS:
    # ;    thresh: stopping criterion for the Newton Raphson iteration; the
    # ;            iteration stops once abs(E-Eold)<thresh
    # ; OUTPUTS:
    # ;    the function returns the eccentric anomaly
    # ; KEYWORD OUTPUT PARAMETERS:
    # ; COMMON BLOCKS:
    # ; SIDE EFFECTS:
    # ; RESTRICTIONS:
    # ; PROCEDURE:
    # ; MODIFICATION HISTORY:
    # ;  2002/05/29 - Marc W. Buie, Lowell Observatory.  Ported from fortran routines
    # ;    supplied by Larry Wasserman and Ted Bowell.
    # ;    http://www.lowell.edu/users/buie/
    # ;
    # ;  2002-09-09 -- Joern Wilms, IAA Tuebingen, Astronomie.
    # ;    use analytical values obtained for the low eccentricity case as
    # ;    starting values for a Newton-Raphson method to allow high
    # ;    eccentricity values as well
    # ;
    # ;  $Log: keplereq.pro,v $
    # ;  Revision 1.3  2005/05/25 16:11:35  wilms
    # ;  speed up: Newton Raphson is only done if necessary
    # ;  (i.e., almost never)
    # ;
    # ;  Revision 1.2  2004/08/05 10:02:05  wilms
    # ;  now also works for more than 32000 time values
    # ;
    # ;  Revision 1.1  2002/09/09 14:54:11  wilms
    # ;  initial release into aitlib
    # ;
    # ;
    # ;-

    #;; set default values
    #IF (n_elements(thresh) EQ 0) THEN thresh=1D-5
    thresh=1e-5

    # ;;
    # ;; Range reduction of m to -pi < m <= pi
    # ;;
    mx=m

    # # ;; ... m > pi
    # zz= np.where(mx > np.pi)[0] #where(mx GT !dpi,count)
    # IF (count NE 0) THEN BEGIN
    #   mx[zz]=mx[zz] MOD (2*!dpi)
    #   zz=where(mx GT !dpi,count)
    #   IF (count NE 0) THEN mx[zz]=mx[zz]-2.0D0*!dpi
    # ENDIF

    mx[mx > np.pi] = mx[mx > np.pi] % (2*np.pi)
    mx[mx < -np.pi] = mx[mx < -np.pi] % (2*np.pi)

    # ;; ... m < -pi
    # zz=where(mx LE -!dpi,count)
    # IF (count NE 0) THEN BEGIN
    #   mx[zz]=mx[zz] MOD (2*!dpi)
    #   zz=where(mx LE -!dpi,count)
    #   IF (count NE 0) THEN mx[zz]=mx[zz]+2.0D0*!dpi
    # ENDIF

    # ;;
    # ;; Bail out for circular orbits...
    # ;;
    # IF (ecc EQ 0.) THEN return,mx
    if ecc == 0:
        return mx
    else:
        return NotImplementedError()
        #
        # aux   =  4.d0*ecc+0.5d0
        # alpha = (1.d0-ecc)/aux
        #
        #
        # beta=mx/(2.d0*aux)
        # aux=sqrt(beta^2+alpha^3)
        #
        # z=beta+aux
        # zz=where(z LE 0.0d0,count)
        # if count GT 0 THEN  z[zz]=beta[zz]-aux[zz]
        #
        # test=abs(z)^0.3333333333333333d0
        #
        # z =  test
        # zz=where(z LT 0.0d0,count)
        # IF count GT 0 THEN z[zz] = -z[zz]
        #
        # s0=z-alpha/z
        # s1=s0-(0.078d0*s0^5)/(1.d0+ecc)
        # e0=mx+ecc*(3.d0*s1-4.d0*s1^3)
        #
        # se0=sin(e0)
        # ce0=cos(e0)
        #
        # f  = e0-ecc*se0-mx
        # f1 = 1.d0-ecc*ce0
        # f2 = ecc*se0
        # f3 = ecc*ce0
        # f4 = -f2
        # u1 = -f/f1
        # u2 = -f/(f1+0.5d0*f2*u1)
        # u3 = -f/(f1+0.5d0*f2*u2+.16666666666667d0*f3*u2*u2)
        # u4 = -f/(f1+0.5d0*f2*u3+.16666666666667d0*f3*u3*u3+.041666666666667d0*f4*u3^3)
        #
        # eccanom=e0+u4
        #
        # zz = where(eccanom GE 2.0d0*!dpi,count)
        # IF count NE 0 THEN  eccanom[zz]=eccanom[zz]-2.0d0*!dpi
        # zz = where(eccanom LT 0.0d0,count)
        # IF count NE 0 THEN eccanom[zz]=eccanom[zz]+2.0d0*!dpi
        #
        # ;; Now get more precise solution using Newton Raphson method
        # ;; for those times when the Kepler equation is not yet solved
        # ;; to better than 1e-10
        # ;; (modification J. Wilms)
        #
        # mmm=mx
        # ndx=where(mmm LT 0.)
        # IF (ndx[0] NE -1 ) THEN BEGIN
        #   mmm[ndx]=mmm[ndx]+2.*!dpi
        # ENDIF
        # diff=eccanom-ecc*sin(eccanom) - mmm
        #
        # ndx=where(abs(diff) GT 1e-10)
        # IF (ndx[0] NE -1) THEN BEGIN
        #   FOR j=0L,n_elements(ndx)-1 DO BEGIN
        #       i=ndx[j]
        #       REPEAT BEGIN
        #           ;; E-e sinE-M
        #           fe=eccanom[i]-ecc*sin(eccanom[i])-mmm[i]
        #           ;; f' = 1-e*cosE
        #           fs=1.-ecc*cos(eccanom[i])
        #           oldval=eccanom[i]
        #           eccanom[i]=oldval-fe/fs
        #       ENDREP UNTIL (abs(oldval-eccanom[i]) LE thresh)
        #       ;; the following should be coded more intelligently ;-)
        #       ;; (similar to range reduction of mx...)
        #       WHILE (eccanom[i] GE  !dpi) DO eccanom[i]=eccanom[i]-2.*!dpi
        #       WHILE (eccanom[i] LT -!dpi ) DO eccanom[i]=eccanom[i]+2.*!dpi
        #   ENDFOR
        # ENDIF
        # return,eccanom
        # END

def cos_f(mean_anom, ecc):
    # ;+
    # ; NAME:
    # ;	cos_f
    # ;
    # ; PURPOSE:
    # ;	This function returns the cosine of the true anomaly, using relations given in
    # ;	Murray & Dermott (1999), ch. 2.4. This routine uses keplereq.pro.
    # ;
    # ; CATEGORY:
    # ;	Celestial Mechanics.
    # ;
    # ; CALLING SEQUENCE:
    # ;	Result = cos_f(mean_anom, ecc)
    # ;
    # ; INPUTS:
    # ;	mean_anom: orbital mean anomaly
    # ;	ecc: orbital eccentricity
    # ;
    # ; OUTPUTS:
    # ;	Cosine of the orbital true anomaly
    # ;
    # ; EXAMPLE:
    # ;       ;make mean anomaly array
    # ;	num = 101
    # ;	mean_anom = 2.*!pi*dindgen(num)
    # ;	;orbital eccentricity of 0.1
    # ;	ecc = 0.1
    # ;	cos_f = cos_f(mean_anom, ecc)
    # ;
    # ; MODIFICATION HISTORY:
    # ; 	Written by:	Brian Jackson (decaelus@gmail.com), 2011 Jul 25.
    # ;-

    #;Call keplereq to calculate eccentric anomaly from mean anomaly
    ecc_anom = keplereq(mean_anom, ecc)
    return (np.cos(ecc_anom)-ecc)/(1.-ecc*np.cos(ecc_anom))

def sin_f(mean_anom, ecc):
    # ;+
    # ; NAME:
    # ;	sin_f
    # ;
    # ; PURPOSE:
    # ;	This function returns the sine of the true anomaly, using relations given in
    # ;	Murray & Dermott (1999), ch. 2.4. This routine uses keplereq.pro.
    # ;
    # ; CATEGORY:
    # ;	Celestial Mechanics.
    # ;
    # ; CALLING SEQUENCE:
    # ;	Result = sin_f(mean_anom, ecc)
    # ;
    # ; INPUTS:
    # ;	mean_anom: orbital mean anomaly
    # ;	ecc: orbital eccentricity
    # ;
    # ; OUTPUTS:
    # ;	Sine of the orbital true anomaly
    # ;
    # ; EXAMPLE:
    # ;       ;make mean anomaly array
    # ;	num = 101
    # ;	mean_anom = 2.*!pi*dindgen(num)
    # ;	;orbital eccentricity of 0.1
    # ;	ecc = 0.1
    # ;	sin_f = sin_f(mean_anom, ecc)
    # ;
    # ; MODIFICATION HISTORY:
    # ; 	Written by:	Brian Jackson (decaelus@gmail.com), 2011 Jul 25.
    # ;-

    # ;Call keplereq to calculate eccentric anomaly from mean anomaly
    ecc_anom = keplereq(mean_anom, ecc)
    return (np.sqrt(1.-ecc**2.)*np.sin(ecc_anom))/(1.-ecc*np.cos(ecc_anom))

