import numpy as np
from iscesys.Component.Component import Component, Port
from isceobj.Orbit.Orbit import Orbit
import isceobj.Util.geo as geo

class Make_los(Component):

    def make_los(self):
        """calculate incidence and squint angles to radar data points"""

        orbit = self._insar.getMasterOrbit()
        r0 = sensor.getFrame().getStartingRange()  # slant range of first pixel
        rsamp = self._insar.getMasterFrame().getInstrument().rangePixelSize
        asamp = self._insar.getAzimuthPixelSize()
        t0 = self._insar.getMasterFrame().getSensingStart()
        prf = self._insar.getMasterFrame().getInstrument().getPulseRepetitionFrequency()
        sc_az_nom = self._insar.getMasterSquint()
        fd_coef = self._insar.getDopplerCentroid().getDopplerCoefficients()
        wvl = self._insar.getMasterFrame().getInstrument().getRadarWavelength()

        # get first pos and vel
        sv = orbit.InterpolateOrbit(t0,method='hermite')
        x,y,z = sv.getPosition()
        pos = geo.WGS84.ECEF(x,y,z)
        vx,vy,vz = sv.getVelocity()
        vel = geo.WGS84.ECEF(vx,vy,vz)

        
        rng = r0 + np.arange(nrmax)*rsamp
        hdg = pos.bearing(pos + vel.hat()*100)
        peg = geo.PegPoint(pos.llh().lat,pos.llh().lon,hdg + sc_az_nom)
        p = pos.sch(peg)
        img_pln_rad = p.ellipsoid.localRad(peg.lat,peg.hdg + hdg)
        trk_rad = p.ellipsoid.localRad(peg.lat,peg.hdg)
        fd_coef_hertz = fd_coef*prf
        spd = float(abs(vel))
        sc_r = float(abs(pos))

        if type_fc == 1:
            rr = rng[nrmax/2]
            pix = (rr - rd_ref)/rsamp_dopp
            dop = np.polyval(fd_coef_hertz,pix)
            th = np.arccos(((p.h + img_pln_rad)**2 + rr**2 - img_pln_rad**2)/(2*rr*(img_pln_rad + p.h)))
            sc_az_nom = lr*(np.pi/2 - np.arcsin(dop*wvl/(2*spd*np.sin(th))))
        else:
            sc_az = np.ones(nrmax)*sc_az_nom

        for i in range(namax):
            nilbuf = np.fromfile(htfile,dtype=float,count=nrmax)
            hgtmap = np.fromfile(htfile,dtype=float,count=nrmax)
            print 'hgtmap',hgtmap
            
            # load new pos,vel orbit state here
            time = t0 + asamp*i*sc_r/(trk_rad*spd)
            sv = orbit.InterpolateOrbit(time,method='hermite')
            x,y,z = sv.getPosition()
            pos = geo.WGS84.ECEF(x,y,z)
            p = pos.llh()
            
            if (type_fc == 1):
                pix = (rng - rd_ref)/rsamp_dopp
                dop = np.polyval(fd_coef_hertz,pix)
                th = np.arccos(((p.h + img_pln_rad)**2 + rr**2 - img_pln_rad**2)/(2*rr*(img_pln_rad + p.h)))
                sc_az = lr*(np.pi/2 - np.arcsin(dop*wvl/(2*spd*np.sin(th))))
                
            target = img_pln_rad + hgtmap
            sc = img_pln_rad + p.h
            look = np.arccos((sc**2 + rng**2 - target**2)/(2*sc*rng))
            beta = np.arccos((sc**2 + target**2 - rng**2)/(2*sc*target))
            inc_angle_rad = look + beta

            inc_angle = np.rad2deg(inc_angle_rad)
            az_angle = np.rad2deg(sc_az) + hdg
            # write out outbuf


        def __init__(self):

            Component.__init__(self)

            self.type_fc = 0
