import sys
import isceobj
import os
import json
import numpy as np
from scipy.ndimage import zoom,gaussian_filter
from .run_unwrap_snaphu import runUnwrap
from iscesys.Component.Component import Component

class DownsampleUnwrapper(Component):
    def __init__(self,inps):
        self._dtype_map = {'cfloat':np.complex64,'float':np.float32,'byte':np.uint8}
        self._ddir = inps['data_dir']
        self._resamp = inps['resamp'] if 'resamp' in inps else 2
        self._cor_name = inps['cor_name'] if 'cor_name' in inps else 'phsig.cor'
        self._unw_name = inps['unw_name'] if 'unw_name' in inps else 'filt_topophase.flat'
        self._flat_name = inps['flat_name'] if 'flat_name' in inps else 'filt_topophase.unw'
        self._ccomp_name =  self._unw_name + '.conncomp'
        self._dcor_name = '{0:s}_{1:d}x{1:d}.cor'.format(self._cor_name.replace('.cor',''),self._resamp)
        self._dunw_name = '{0:s}_{1:d}x{1:d}.unw'.format(self._unw_name.replace('.unw',''),self._resamp)
        self._dflat_name = '{0:s}_{1:d}x{1:d}.flat'.format(self._flat_name.replace('.flat',''),self._resamp)
        self._dccomp_name =  self._dunw_name + '.conncomp'
        self._earth_radius = inps['earth_radius'] if 'earth_radius' in inps else 6371000
        self._altitude = inps['altitude'] if 'altitude' in inps else 800000
        self._wavelength = inps['wavelength'] if 'wavelength' in inps else 0
        self._azimuth_looks = inps['azimuth_looks']
        self._range_looks = inps['range_looks']
        self._remove_downsampled = True if 'remove_downsampled' in inps else False
        
    def get_isce_image(self,itype,fname,width,length):
        if itype == 'flat':
            im = isceobj.createIntImage()
        else:
            im = isceobj.createImage()
            im.dataType = 'FLOAT'                
            if itype == 'unw':
                im.bands = 2
                im.scheme = 'BIL'
            elif itype == 'ccomp':
                im.dataType = 'BYTE'
        im.filename = fname
        im.extraFilename = fname + '.vrt'
        im.width = width
        im.length = length
        return im

    def save_image(self,ddir,fname,img,itype):
        dname = os.path.join(ddir,fname)
        im = self.get_isce_image(itype,dname,img.shape[-1],img.shape[0])
        img.astype(self._dtype_map[im.dataType.lower()]).tofile(dname)
        im.dump(dname + '.xml')
        
    def remove_downsampled(self,ddir,flat,unw,phsig,ccomp):
        try:
            #subprocess keeps changing API. just use system
            cmd = 'rm -rf {}* {}* {}* {}*'.format(os.path.join(ddir,flat),os.path.join(ddir,unw),os.path.join(ddir,phsig),os.path.join(ddir,ccomp))
            os.system(cmd)
        except Exception as e:
            print(e)
            
    def downsample_images(self,ddir,flat,phsig,resamp):
        img,im = self.load_image(ddir,flat)
        _,co = self.load_image(ddir,phsig)
        ims = []
        cos = []
        width = img.width
        length = img.length
        width -= width%resamp
        length -= length%resamp
        for i in range(resamp):
            for j in range(resamp):
                ims.append(im[i:length:resamp,j:width:resamp])
                cos.append(co[i:length:resamp,j:width:resamp])
        ims = np.array(ims).transpose([1,2,0])
        cos = np.array(cos).transpose([1,2,0])
        nims = ims.mean(2)
        ncos = cos.mean(2)
        self.save_image(ddir, self._dcor_name,ncos,'cor')    
        self.save_image(ddir, self._dflat_name,nims,'flat')    

    
    def load_image(self,ddir,fname):
        img = isceobj.createImage()
        img.load(os.path.join(ddir,fname + '.xml'))
        dtype = self._dtype_map[img.dataType.lower()]
        width = img.getWidth()
        length = img.getLength()
        im = np.fromfile(os.path.join(ddir,fname),dtype)
        if img.bands == 1:
            im = im.reshape([length,width])
        else:#the other option is the unw which is 2 bands BIL
            im = im.reshape([length,img.bands,width])
        return img,im
      
    def upsample_unw(self,ddir,flat,unw,ccomp,upsamp=2,filt_sizes=(3,4)):
        _,dunw = self.load_image(ddir,unw)
        _,flati = self.load_image(ddir,flat)
        _,dccomp = self.load_image(ddir,ccomp)
        uccomp = zoom(dccomp,upsamp)
        uccomp = np.round(uccomp).astype(np.uint8)
        ph_unw = dunw[:,1,:]
        amp = np.abs(flati)
        ph_flat = np.angle(flati)
        uph_unw = zoom(ph_unw,upsamp)
        uph_size = uph_unw.shape
        ph_size = ph_flat.shape 
        if uph_size[0] != ph_size[0] or uph_size[0] != ph_size[1]:
            #the lost one pixel during downsampling/upsampling
            nunw = np.zeros(ph_flat.shape)
            nunw[:uph_size[0],:uph_size[1]] = uph_unw
            if ph_size[1] > uph_size[1]:
                nunw[-1,:-1] = uph_unw[-1,:]
            if ph_size[0] > uph_size[0]:
                nunw[:-1,-1] = uph_unw[:,-1] 
            uph_unw = nunw
        funw = self.filter_image(uph_unw,ph_flat,filt_sizes)
        ifunw = np.round((funw - ph_flat)/(2*np.pi)).astype(np.int16)
        funw = ph_flat + 2*np.pi*ifunw
        unw_out = np.stack([amp,funw],0).transpose([1,0,2])
        self.save_image(ddir,self._unw_name,unw_out,'unw')
        self.save_image(ddir,self._ccomp_name,uccomp,'ccomp')
        if self._remove_downsampled:
            self.remove_downsampled(ddir, self._dflat_name,self._dunw_name,self._dccomp_name,self._dccomp_name)
    
    def filter_image(self,unw,wrp,filt_sizes):
        im0 = np.round((unw - wrp)/(2*np.pi))
        img = wrp + 2*np.pi*im0
        for filter in filt_sizes:
            if not isinstance(filter,tuple):
                filter = (filter,filter)
            img = gaussian_filter(img,filter,0)
        return wrp + 2*np.pi*np.round((img - wrp)/(2*np.pi))
    
    def run_snaphu(self):
        range_looks = int(self._range_looks*self._resamp)
        azimuth_looks = int(self._azimuth_looks*self._resamp)
        inps_json = {'flat_name':self._dflat_name,'unw_name':self._dunw_name,
                     'cor_name':self._dcor_name,'wavelength':self._wavelength,
                     'range_looks':range_looks,'azimuth_looks':azimuth_looks,
                     'earth_radius':self._earth_radius,'altitude':self._altitude}
        runUnwrap(inps_json) 
    
def main(inps):
    """
    Run the unwrapper with downsampling
    inputs:
        inps = dictionary with the following key,value
        {
            "flat_name":"filt_topophase.flat",
            "unw_name":"filt_topophase.unw",
            "cor_name":"phsig.cor",
            "range_looks":7,
            "azimuth_looks":3,
            "wavelength":0.05546576,
            "resamp":2,
            "data_dir":'./'
        }
            The range and azimuth looks are w.r.t the original image.
    """
    du = DownsampleUnwrapper(inps)
    #du.downsample_images(du._ddir,du._flat_name,du._cor_name,du._resamp)
    #du.run_snaphu()
    du.upsample_unw(du._ddir,du._flat_name,du._dunw_name,du._dccomp_name,upsamp=du._resamp,filt_sizes=(3,4))
    

if __name__ == '__main__':
    inp_json = sys.argv[1]
    ddir = sys.argv[2]
    inps = json.load(open(inp_json))
    inps['data_dir'] = ddir
    main(inps)
    
    