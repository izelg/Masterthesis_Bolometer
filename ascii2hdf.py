#%%
# Needed package: pyhdf
from pyhdf import SD
import numpy as np

#%%
def _hdf_writearr(sd, name, array,*,
                 MINIMUM_DIM=None, CREATE_DIMS=None, 
                 KEEP_EXISTING=None, DIMLABELS=None, **_EXTRA):

    # Treatment of extra keywords
    arglist = []; argdict = {}
    for k,v in _EXTRA.items():
        if k in ["start", "count", "stride"]:
            arglist.append((k, v))
        argdict = dict(arglist)

    # Output
    if len(array) < 40: 
        print("hdf_writearr: "+name+" ", array)
    else: 
        print("hdf_writearr: "+name)

    # Add degenerate trailing dimensions if too few dimensions
    md = 0 if MINIMUM_DIM == None else MINIMUM_DIM

    # see if sds already exists
    sds = -1
    if KEEP_EXISTING != None:
        number = sd.nametoindex(name)
        if number >= 0: sds = sd.select(number)
        else: print("hdf_writearr error: "+name+" does not exist, creating")
    
    if all(isinstance(elem, str) for elem in array): # special treatment for strings
        sarr = [s.replace('\x00', ' ') for s in array]
        barr = []
        for i in range(len(sarr)):
            ssub = sarr[i]
            barr = barr + [[ord(ch) for ch in ssub]]
        narr = np.array(barr, dtype=np.uint8)
        asize = CREATE_DIMS if CREATE_DIMS != None else np.shape(narr) 
        print("dims ", asize)
        if sds == -1: sds = sd.create(name, SD.SDC.UINT8, asize)
        #if sds <= 0:
        #    print("hdf_writearr error: "+name+" not created")
        #    return -1

        if argdict: sds.set(narr, argdict)#, _EXTRA=_EXTRA)
        else: sds.set(narr)
        
    else:
        farr = array
        tTyp = SD.SDC.CHAR if (farr.dtype == np.byte) else \
               SD.SDC.CHAR8 if (farr.dtype == np.byte) else \
               SD.SDC.UCHAR8 if (farr.dtype == np.ubyte) else \
               SD.SDC.INT8 if (farr.dtype == np.int8) else \
               SD.SDC.UINT8 if (farr.dtype == np.uint8) else \
               SD.SDC.INT16 if (farr.dtype == np.int16) else \
               SD.SDC.UINT16 if (farr.dtype == np.uint16) else \
               SD.SDC.INT32 if (farr.dtype == np.int32) else \
               SD.SDC.UINT32 if (farr.dtype == np.uint32) else \
               SD.SDC.FLOAT32 if (farr.dtype == np.float32) else \
               SD.SDC.FLOAT64 if (farr.dtype == np.float64) else \
               SD.SDC.UINT8
        # promote scalars to array
        ndims = len(np.shape(farr))
        if ndims == 0: farr = [farr]
        if ndims < md: farr.reshape(*farr.shape,1)
        asize = CREATE_DIMS if CREATE_DIMS != None else np.shape(farr) 
        print("dims ", asize)
        if sds == -1: sds = sd.create(name, tTyp, asize) 
        #if sds <= 0:
        #    print("hdf_writearr error: "+name+" not created")
        #    return -1

        if argdict: sds.set(farr, argdict)#, _EXTRA=_EXTRA)
        else: sds.set(farr)

    if DIMLABELS != None:
        for di in range(len(DIMLABELS)):
            Result = sds.dim(di)
            Result.setname(DIMLABELS[di])

    sds.endaccess()

    return sds






def _hdf_readarr(sd, name, array, *,
                MINIMUM_DIM=None, DIMLABELS=None, **_EXTRA):
    
    # Treatment of extra keywords
    arglist = []; argdict = {}
    for k,v in _EXTRA.items():
        if k in ["start", "count", "stride"]:
            arglist.append((k, v))
        argdict = dict(arglist)

    # get dataset by name
    number = sd.nametoindex(name)
    if number < 0:
        print("hdf_readarr: "+name+" not found")
        return -1
    sds = sd.select(number)
    tarray = sds.get() if not argdict else sds.get(argdict)#_EXTRA=_EXTRA)

    if DIMLABELS != None:
        _sa = len(np.shape(tarray))
        if _sa == 0: _sa = 1
        DIMLABELS = []
        for di in range(_sa):
            Result = sds.dim(di)
            label = Result.info()[0] 
            DIMLABELS.append(label)

    sds.endaccess()

    # promote scalars to array
    ndims = len(np.shape(tarray))
    if ndims == 0: tarray = tarray.reshape(*tarray.shape + 1)
    
    # add degenerate trailing dimensions if too few dimensions
    md = 0 if MINIMUM_DIM == None else MINIMUM_DIM
    
    ndims = len(np.shape(tarray))
    if ndims < md: tarray = tarray.reshape(*tarray.shape + 
                                           tuple(np.ones(md-ndims, dtype=np.int32)))
    array.clear()
    array.append(tarray)
    
    return sds



            

def _hdf_readarrstr(sd, name, array,
                   MINIMUM_DIM=None, DIMLABELS=None, **_EXTRA):
    
    md = 0 if MINIMUM_DIM == None else MINIMUM_DIM+1
    tarray = []
    asdf = _hdf_readarr(sd, name, tarray, 
                MINIMUM_DIM=md, DIMLABELS=DIMLABELS, _EXTRA=_EXTRA)
    array.clear()
    tbuffer = []
    for i in range(np.shape(tarray[0])[0]):
        tbuffer = tbuffer + [str(tarray[0][i].tostring(), 'utf-8')]
    
    array.clear()
    array.append(tbuffer)

    return asdf
    
    


class hdf_fft:
    def __init__(self):
        self.freqs = []
        self.fftd = []
        self.phases = []

class hdf_fileinfo:
    def __init__(self):
        self.coords = []
        self.coordsoffset = []
        self.probel = []
        self.prober = []
        self.resistors = []
        self.scale = []
        self.srate = []
        self.probedims = []
        self.channspec = []
        self.comments = []
        self.devicenum = []
        self.inplimit = []

class hdf_waveforms:
    def __init__(self):
        self.waveforms = []



    

def hdf_readffts(filename, hdf_fft):
    sd = SD.SD(filename, SD.SDC.READ)
    
    freqs = []; fftd = []; phases = []
    
    asdf = hdf_readarr(sd, "frequencies", freqs, MINIMUM_DIM=3)
    asdf = hdf_readarr(sd, "powerspectrum", fftd, MINIMUM_DIM=3)
    asdf = hdf_readarr(sd, "phases", phases, MINIMUM_DIM=3)   
    
    sd.end()
    
    hdf_fft.freqs = freqs[0]
    hdf_fft.fftd = fftd[0]
    hdf_fft.phases = phases[0]

    


def hdf_writewaveform(filename, hdf_fileinfo, hdf_waveforms):

    sd = SD.SD(filename, SD.SDC.WRITE|SD.SDC.CREATE)
 
    numch = np.shape(hdf_waveforms.waveforms)[0]
    asdf = _hdf_writearr( sd, "coords", hdf_fileinfo.coords, MINIMUM_DIM=2, DIMLABELS=["channel no", "coordinate no"]) 
    asdf = _hdf_writearr( sd, "coords offset", hdf_fileinfo.coordsoffset, MINIMUM_DIM=2, DIMLABELS=["channel no", "coordinate no"]) 
    asdf = _hdf_writearr( sd, "device num", hdf_fileinfo.devicenum, DIMLABELS=[ "channel no"]) 
    asdf = _hdf_writearr( sd, "input limits", hdf_fileinfo.inplimit, MINIMUM_DIM=2, DIMLABELS=["channel no", "low, high"]) 
    asdf = _hdf_writearr( sd, "probe dimensions", hdf_fileinfo.probedims, MINIMUM_DIM=2, DIMLABELS=["channel no", "dimension no"]) 
    asdf = _hdf_writearr( sd, "resistors", hdf_fileinfo.resistors, DIMLABELS=[ "channel no"]) 
    asdf = _hdf_writearr( sd, "scale", hdf_fileinfo.scale, DIMLABELS=[ "channel no"]) 
    asdf = _hdf_writearr( sd, "scan rate", hdf_fileinfo.srate) 
    asdf = _hdf_writearr( sd, "channel spec", hdf_fileinfo.channspec, DIMLABELS=["channel no"]) 
    asdf = _hdf_writearr( sd, "comments", hdf_fileinfo.comments, DIMLABELS=["channel no"]) 
    asdf = _hdf_writearr( sd, "waveforms", hdf_waveforms.waveforms, MINIMUM_DIM=2, DIMLABELS=["channel no", "time index"])

    sd.end()




def hdf_readwaveform(filename, hdf_fileinfo, hdf_waveforms):

    sd = SD.SD(filename, SD.SDC.READ)
    
    dm = []; coords = []; coordsoff = []; devicenum = []; 
    inplimit = []; probedims = []; resistors = []; scale = [];
    tscanrate = []; channspec = []; comments = [];
    
    asdf = _hdf_readarr( sd, "coords", coords, MINIMUM_DIM=2, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "coords offset", coordsoff, MINIMUM_DIM=2, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "device num", devicenum, MINIMUM_DIM=1, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "input limits", inplimit, MINIMUM_DIM=2, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "probe dimensions", probedims, MINIMUM_DIM=2, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "resistors", resistors, MINIMUM_DIM=1, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "scale", scale, MINIMUM_DIM=1, DIMLABELS=dm) 
    asdf = _hdf_readarr( sd, "scan rate", tscanrate, MINIMUM_DIM=2, DIMLABELS=dm)
    scanrate = tscanrate[0]
    asdf = _hdf_readarrstr( sd, "channel spec", channspec, MINIMUM_DIM=1, DIMLABELS=dm) 
    asdf = _hdf_readarrstr( sd, "comments", comments, MINIMUM_DIM=1, DIMLABELS=dm)
    
    # LabView does not fill comments up to num_channels (?)
    nco = len( comments[0]) 
    nch = len( scale[0])
    if nco < nch: comments[0] = comments[0] + [''] * (nch-nco)

    # coords need not be supplied
    if not coords: coords = [np.zeros([2, nch], dtype=np.float32)]
    if not coordsoff: coordsoff = [np.zeros([2, nch], dtype=np.float32)]

    waveforms = []
    asdf = _hdf_readarr( sd, "waveforms", waveforms, DIMLABELS=dm, MINIMUM_DIM=2)
    #nump = np.shape(waveforms[0])[0]

    sd.end()

    hdf_fileinfo.coords = coords[0]
    hdf_fileinfo.coordsoffset = coordsoff[0]
    hdf_fileinfo.probel = (probedims[0])[1,:]
    hdf_fileinfo.prober = (probedims[0])[0,:]
    hdf_fileinfo.resistors = resistors[0]
    hdf_fileinfo.scale = scale[0]
    hdf_fileinfo.srate = scanrate[0]
    hdf_fileinfo.probedims = probedims[0]
    hdf_fileinfo.channspec = channspec[0]
    hdf_fileinfo.comments = comments[0]
    hdf_fileinfo. devicenum = devicenum[0]
    hdf_fileinfo. inplimit = inplimit[0]
    
    hdf_waveforms.waveforms = waveforms[0]


# %%
shotnumber=13037
hdf_path='/data6/shot{}/probe2D/data000000.hdf'.format(shotnumber)
hdf_readffts(hdf_path,hdf_path)



# %%
