import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import galsim
import galsim.des
import galsim.config
import math
import os, sys
import os.path

class Gazou:
    
    """ This class creates an instance for images, including the main image and the segmentation map,
        plus an instance for catalogues, including the Source Extractor one and the PSFEx catalgoue. """
    
    def __init__(self, tile_img_name, tile_segmap_name, catalogue_name, psf_catalogue_name):
        
        # Check the dimension of the wcs system
        dim_img = 0
        hdulist_tile = fits.open(tile_img_name)
        header_tile = hdulist_tile[0].header
        if 'NAXIS1' in header_tile:
            dim_img = dim_img + 0
        else:
            dim_img = dim_img + 1
        
        dim_seg = 0
        hdulist_segmap = fits.open(tile_segmap_name)
        header_segmap = hdulist_segmap[0].header
        if 'NAXIS1' in header_segmap:
            dim_seg = dim_seg + 0
        else:
            dim_seg = dim_seg + 1
            
        # Read and store information on the main image
        hdulist_tile = fits.open(tile_img_name)
        data_tile = hdulist_tile[dim_img].data
        header_tile = hdulist_tile[dim_img].header
        wcs_tile = wcs.WCS(header_tile)
        
        self.tile_img_name = tile_img_name
        self.data_tile = data_tile
        self.header_tile = header_tile
        self.wcs_tile = wcs_tile
        
        # Read and store information on the main segmentation map
        hdulist_segmap = fits.open(tile_segmap_name)
        data_segmap = hdulist_segmap[dim_seg].data
        header_segmap = hdulist_segmap[dim_seg].header
        wcs_segmap = wcs.WCS(header_segmap)
        
        self.data_segmap = data_segmap
        self.header_segmap = header_segmap
        self.wcs_segmap = wcs_segmap
        
        # Read and store information on the .fits catalogue
        pha_list = fits.open(catalogue_name)
        self.catalogue = pha_list[1].data
        
        # Read and store information on the PSEFEx catalogue
        self.psf_catalogue_name = psf_catalogue_name
        self.psf_catalogue = galsim.des.DES_PSFEx(psf_catalogue_name, tile_img_name)
        
        # Setup the keys for mag_psf and mag_disk, according to their availability in the catalogue
        keys = self.catalogue.columns.names
        if 'MAG_PSF' in keys:
            self.mag_psf_key = 'MAG_PSF'
        else:
            self.mag_psf_key = 'MAG_AUTO'
            
        if 'MAG_DISK' in keys:
            self.mag_disk_key = 'MAG_DISK'
        else:
            self.mag_disk_key = 'MAG_AUTO'
    
    
    def add_params(self, psf_img_size, pixel_scale, magzeropoint, exptime, ncombine):
        
        ''' This function allows for implementing additional attributes which will be used 
        during the pipeline '''
        
        self.psf_img_size = psf_img_size
        self.pixel_scale = pixel_scale
        self.magzeropoint = magzeropoint
        self.header_tile['EXPTIME'] = exptime
        self.header_tile['NCOMBINE'] = ncombine
        
        print('Creating stamps with mag_zeropoint =', self.magzeropoint, ', EXPTIME =', self.header_tile['EXPTIME'], 
             ', NCOMBINE =', self.header_tile['NCOMBINE'], ', pixel scale =', self.pixel_scale, '.')
        print('Size of the PSF stamp =', self.psf_img_size, '.')
        
        
    def add_paths(self, stamps_folder, galfit_folder):
        
        ''' This function allows for adding as attributes the paths where to store the produced images '''
        
        folder_img = 'Img/'
        folder_seg = 'Seg/'
        folder_cleaned_seg = 'Cleaned_Seg/'
        folder_PSF = 'PSF/'
        
        self.img_output_folder = os.path.join(stamps_folder + folder_img) 
        self.seg_output_folder = os.path.join(stamps_folder + folder_seg) 
        self.clean_seg_output_folder = os.path.join(stamps_folder + folder_cleaned_seg) 
        self.psf_output_folder = os.path.join(stamps_folder + folder_PSF)
            
        os.makedirs(self.img_output_folder, exist_ok = True)
        os.makedirs(self.seg_output_folder, exist_ok = True)
        os.makedirs(self.clean_seg_output_folder, exist_ok = True)
        os.makedirs(self.psf_output_folder, exist_ok = True)
        
        folder_psf_sersic_fit = 'PSF_Sersic/'
        folder_psf_disk_fit = 'PSF_Disk/'
        folder_psf_double_sersic_fit = 'PSF_Double_Sersic/'
        folder_sersic_fit = 'Sersic/'
        folder_bulge_disk_fit = 'BulgeDisk/'
        folder_bulge_disk_41_fit = 'BulgeDisk_41/'
        folder_double_sersic_fit = 'Double_Sersic/'
        
        self.psf_sersic_folder = os.path.join(galfit_folder + folder_psf_sersic_fit)
        self.psf_double_sersic_folder = os.path.join(galfit_folder + folder_psf_double_sersic_fit)
        self.psf_disk_folder = os.path.join(galfit_folder + folder_psf_disk_fit)
        self.sersic_folder = os.path.join(galfit_folder + folder_sersic_fit)
        self.bulge_disk_folder = os.path.join(galfit_folder + folder_bulge_disk_fit)
        self.bulge_disk_41_folder = os.path.join(galfit_folder + folder_bulge_disk_41_fit)
        self.double_sersic_folder = os.path.join(galfit_folder + folder_double_sersic_fit)
        
        os.makedirs(self.psf_sersic_folder, exist_ok = True)
        os.makedirs(self.psf_double_sersic_folder, exist_ok = True)
        os.makedirs(self.psf_disk_folder, exist_ok = True)
        os.makedirs(self.sersic_folder, exist_ok = True)
        os.makedirs(self.bulge_disk_folder, exist_ok = True)
        os.makedirs(self.bulge_disk_41_folder, exist_ok = True)
        os.makedirs(self.double_sersic_folder, exist_ok = True)
        
        Gazou._add_galfit_constr(self)
        Gazou._add_rungalfit(self)
            
            
    def _add_galfit_constr(self):
        
        self.psf_sersic_constr = '../../ConstraintsFiles/psf_sersic.CONSTR'
        self.psf_disk_constr = '../../ConstraintsFiles/psf_disk.CONSTR'
        self.psf_double_sersic_constr = '../../ConstraintsFiles/psf_double_sersic.CONSTR'
        self.sersic_constr = '../../ConstraintsFiles/sersic.CONSTR'
        self.bulge_disk_constr = '../../ConstraintsFiles/bulge_disk.CONSTR'
        self.bulge_disk_41_constr = '../../ConstraintsFiles/bulge_disk_41.CONSTR'
        self.double_sersic_constr = '../../ConstraintsFiles/double_sersic.CONSTR'
        
    def _add_rungalfit(self):
        
        os.system("cp Galfit_Scripts/rungalfit.py " + self.psf_sersic_folder)
        os.system("cp Galfit_Scripts/rungalfit.py " + self.psf_double_sersic_folder)
        os.system("cp Galfit_Scripts/rungalfit.py " + self.psf_disk_folder)
        os.system("cp Galfit_Scripts/rungalfit.py " + self.sersic_folder)
        os.system("cp Galfit_Scripts/rungalfit.py " + self.bulge_disk_folder)
        os.system("cp Galfit_Scripts/rungalfit.py " + self.bulge_disk_41_folder)
        os.system("cp Galfit_Scripts/rungalfit.py " + self.double_sersic_folder)
        
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.psf_sersic_folder)
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.psf_double_sersic_folder)
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.psf_disk_folder)
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.sersic_folder)
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.bulge_disk_folder)
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.bulge_disk_41_folder)
        os.system("cp Galfit_Scripts/multicores_rungalfit.py " + self.double_sersic_folder)
           
    def create_stamps(self, obj):
        
        ''' This function manages the following operations:
            - i) collecting catalogue-information about the selected object;
            - ii) creation of the stamps (images and segmentation maps);
            - iii) extraction of the psf for each galaxy;
            - iv) cleaning the postage stamps from distant sources;
            - v) creating the .feedme files which will be used to (Gal)fit the images. '''
        
        self.obj_id = obj
        
        # i) Collecting catalogue-information about the selected object
        self.NUMBER = self.catalogue['NUMBER'][obj]
        self.ra = self.catalogue['ALPHAPEAK_J2000'][obj]
        self.dec = self.catalogue['DELTAPEAK_J2000'][obj]
        self.x = float(self.catalogue['X_IMAGE'][obj])
        self.y = float(self.catalogue['Y_IMAGE'][obj])
        
        self.b_image = float(self.catalogue['B_IMAGE'][obj])
        self.radius = float(self.catalogue['FLUX_RADIUS'][obj])
        self.sky = float(self.catalogue['BACKGROUND'][obj])
        self.mag_auto = float(self.catalogue['MAG_AUTO'][obj])
        self.mag_psf = float(self.catalogue[self.mag_psf_key][obj])
        self.mag_disk = float(self.catalogue[self.mag_disk_key][obj])
        self.theta = float(self.catalogue['THETA_IMAGE'][obj] - 90)
        
        self.a_image = float(self.catalogue['A_IMAGE'][obj])
        self.kr = float(self.catalogue['KRON_RADIUS'][obj])
        
        self.input_size = round(self.kr * self.a_image,0) * 6
        if self.input_size % 2 == 0:
            self.input_size = self.input_size - 1
        self.wd = self.input_size * 1./2
        
        # ii) creation of the stamps anf the relative segmentation maps
        self.img = Gazou._canvas(self, self.data_tile, self.wcs_tile)
        self.seg = Gazou._canvas(self, self.data_segmap, self.wcs_segmap)
        
        nameout_img = self.img_output_folder + 'img_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        nameout_seg = self.seg_output_folder + 'seg_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        
        newfile_img = fits.PrimaryHDU(data = self.img[0], header = self.img[1])
        newfile_img.writeto(nameout_img, overwrite=True, checksum=True)
        
        newfile_seg = fits.PrimaryHDU(data = self.seg[0], header = self.seg[1])
        newfile_seg.writeto(nameout_seg, overwrite=True, checksum=True)
        
        
        # iii) psf extraction
        Gazou._get_psf(self)
        self.nameout_psf = self.psf_output_folder + 'psf_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        self.name_psf_galfit = '../../' + self.nameout_psf
        
        # iv) cleaning segmentation maps and preparation for Galfit
        self.nameout_clean = self.clean_seg_output_folder + 'seg_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        self.seg[0][self.seg[0]==self.NUMBER] = 0
        newfile_clean = fits.PrimaryHDU(data = self.seg[0], header = self.seg[1])
        newfile_clean.writeto(self.nameout_clean, overwrite = True, checksum = True)
        
        self.name_img_galfit = '../../' + nameout_img
        self.name_mask_galfit = '../../' + self.nameout_clean
        
        # v) creating the .feedme files which will be used for (Gal)fit the images.
        self.imcut_dim = self.img[1]['NAXIS1']
        self.sky = round(self.sky,4)
        self.mag_auto = round(self.mag_auto,4)
        self.mag_psf = round(self.mag_psf,4)
        self.mag_disk = round(self.mag_disk,4)
        self.radius = round(self.radius,4)
        self.ar = round(self.b_image/self.a_image,4)
        self.theta = round(self.theta, 4)
        
        self.conv_box = self.imcut_dim + 2
        self.xy_position = (self.imcut_dim+1)/2
        
        self.radius_max = np.round(self.kr*self.a_image,1)
        
        Gazou._psf_sersic(self)
        Gazou._single_sersic(self)
        Gazou._bulge_disk(self)
        Gazou._bulge_disk_41(self)
        Gazou._double_sersic(self)
        Gazou._psf_disk(self)
        Gazou._psf_double_sersic(self)
        
        
    def _canvas(self, image_matrix, w):
        
        xx,yy = w.wcs_world2pix(self.ra, self.dec, 1)
        xw = self.wd
        yw = self.wd
        
        head = self.header_tile.copy()
        
        xmin,xmax = np.int(np.max([0,xx-xw])),np.int(np.min([head['NAXIS1'],xx+xw]))
        ymin,ymax = np.int(np.max([0,yy-yw])),np.int(np.min([head['NAXIS2'],yy+yw]))
        
        # CRPIX = 2 element vector giving X and Y coordinates of reference pixel (def = NAXIS/2) in FITS convention (first pixel is 1,1)
        head['CRPIX1']-=xmin
        head['CRPIX2']-=ymin
        head['NAXIS1']=xmax-xmin
        head['NAXIS2']=ymax-ymin
        
        head['KR'] = self.kr
        head.comments['KR'] = 'Source Extractor Kron Radius' 
        head['A_IMAGE'] = self.a_image
        head.comments['A_IMAGE'] = 'Source Extractor Major Axis [px]'
        head['B_IMAGE'] = self.b_image
        head.comments['B_IMAGE'] = 'Source Extractor minor axis [px]'
        head['FLUX_RAD'] = self.radius
        head.comments['FLUX_RAD'] = 'Source Extractor Flux Radius [px]'
        head['BKG'] = self.sky
        head.comments['BKG'] = 'Source Extractor Background'
        head['MAG_AUTO'] = self.mag_auto
        head.comments['MAG_AUTO'] = 'Source Extractor Mag Auto'
        head['MAG_PSF'] = self.mag_psf
        head.comments['MAG_PSF'] = 'Source Extractor Mag PSF'
        head['MAG_DISK'] = self.mag_disk
        head.comments['MAG_DISK'] = 'Source Extractor Mag Disk'
        head['THETA'] = self.theta
        head.comments['THETA'] = 'Source Extractor Theta Image - 90 (degree)'
        
        head['NUMBER'] = self.NUMBER
        head.comments['NUMBER'] = 'Source Extractor NUMBER'
        head['RA'] = self.ra
        head.comments['RA'] = 'Source Extractor ALPHAPEAK_J2000  (wcs degree)'
        head['DEC'] = self.dec
        head.comments['DEC'] = 'Source Extractor DELTAPEAK_J2000  (wcs degree)'
        head['X'] = self.x
        head.comments['X'] = 'Source Extractor X_IMAGE [px]'
        head['Y'] = self.y
        head.comments['Y'] = 'Source Extractor Y_IMAGE [px]'
        
        
        img = image_matrix[ymin:ymax,xmin:xmax]
        return (img,head)
    
    
    def _get_psf(self):
        
        # Position in pixels on the image, not in arcsec on the sky! 
        # The profile is in world coordinates
        image_pos = galsim.PositionD(self.x, self.y)  
        
        des_psfex = galsim.des.DES_PSFEx(self.psf_catalogue_name, self.tile_img_name)
        psf_img = des_psfex.getPSF(image_pos)
        
        im = psf_img.drawImage()
        
        naxes = int(self.psf_img_size)
        full_image = galsim.Image(naxes, naxes, scale = int(self.pixel_scale))
        b = galsim.BoundsI(1, naxes, 1, naxes)
        stamp = psf_img.drawImage(image = full_image[b])
        assert (stamp.array == full_image[b].array).all()
        self.nameout_psf = self.psf_output_folder + 'psf_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        stamp.write(self.nameout_psf)
            
        
    def _single_sersic(self):
        
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.sersic_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        feedme_name = self.sersic_folder + 'sersic_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        
        f.writelines(['\n',
                      '===============================================================================\n',
                      '# IMAGE and GALFIT CONTROL PARAMETERS\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 2.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
     
    
    def _bulge_disk(self):
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.bulge_disk_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    1              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1_2            x        ratio        # forcing components to share the same center\n',
                      '    1_2            y        ratio        # forcing components to share the same center\n',
                      '    2              rs        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        
        feedme_name = self.bulge_disk_folder + 'bulgedisk_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        
        f.writelines(['\n',
                      '===============================================================================\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 4.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) expdisk                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_disk) + '     1          #  Integrated magnitude	\n',
                      ' 4) ' + str(round(self.radius/1.678,4)) + '     1          #  R_s (disk scale-length)   [pix]\n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      'C0) -0.05       1                    #  diskyness(-)/boxyness(+)'
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number 3\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
        
    
    def _bulge_disk_41(self):
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.bulge_disk_41_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              n        -1  1        # Sérsic index n to be in the interval [3, 5]\n',
                      '    1              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1_2            x        ratio        # forcing components to share the same center\n',
                      '    1_2            y        ratio        # forcing components to share the same center\n',
                      '    2              rs        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        feedme_name = self.bulge_disk_41_folder + 'bulgedisk_41_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        
        f.writelines(['\n',
                      '===============================================================================\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 4.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) expdisk                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_disk) + '     1          #  Integrated magnitude	\n',
                      ' 4) ' + str(round(self.radius/1.678,4)) + '     1          #  R_s (disk scale-length)   [pix]\n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      'C0) -0.05       1                    #  diskyness(-)/boxyness(+)'
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number 3\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
        
        
    def _double_sersic(self):
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.double_sersic_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    1              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '    2              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1_2            x        ratio        # forcing components to share the same center\n',
                      '    1_2            y        ratio        # forcing components to share the same center\n',
                      '    2              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        feedme_name = self.double_sersic_folder + 'double_sersic_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        
        f.writelines(['\n',
                      '===============================================================================\n',
                      '# IMAGE and GALFIT CONTROL PARAMETERS\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 4.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 2.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 3\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
                     
                     
    def _psf_disk(self):
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.psf_disk_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1_2            x        ratio        # forcing components to share the same center\n',
                      '    1_2            y        ratio        # forcing components to share the same center\n',
                      '    2              rs        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        feedme_name = self.psf_disk_folder + 'psf_disk_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        f.writelines(['\n',
                      '===============================================================================\n',
                      '# IMAGE and GALFIT CONTROL PARAMETERS\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) psf                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '   1 1   #  position x, y\n',
                      ' 3) ' + str(self.mag_psf) + '     1          #  Integrated magnitude    \n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) expdisk                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_disk) + '     1          #  Integrated magnitude	\n',
                      ' 4) ' + str(round(self.radius/1.678,4)) + '     1          #  R_s (disk scale-length)   [pix]\n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      'C0) -0.05       1                    #  diskyness(-)/boxyness(+)'
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 3\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
        
        f.close()

        
    def _psf_sersic(self):
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.psf_sersic_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1_2            x        ratio        # forcing components to share the same center\n',
                      '    1_2            y        ratio        # forcing components to share the same center\n',
                      '    2              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    2              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        feedme_name = self.psf_sersic_folder + 'psf_sersic_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        f.writelines(['\n',
                      '===============================================================================\n',
                      '# IMAGE and GALFIT CONTROL PARAMETERS\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) psf                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '   1 1   #  position x, y\n',
                      ' 3) ' + str(self.mag_psf) + '     1          #  Integrated magnitude    \n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 2.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 3\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
        
        f.close()
        
        
    def _psf_double_sersic(self):
        
        outputname = 'fit_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.fits'
        constr_name = 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        constr_file = self.psf_double_sersic_folder + 'file_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.CONSTR'
        c = open(constr_file, 'w')
        c.writelines(['# Component    parameter    constraint    Comment\n',
                      '    1              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    2              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    3              x        -1  1        # x-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    3              y        -1  1        # y-position to be within +1 and -1 of the >>INPUT<< value\n',
                      '    1_2_3            x        ratio        # forcing components to share the same center\n',
                      '    1_2_3            y        ratio        # forcing components to share the same center\n',
                      '    3              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    3              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '    2              n        0.7 to 10    # Sérsic index n to be in the interval [0.7, 10]\n',
                      '    2              re        0.5 to ' + str(self.radius_max) + '    # Effective radius Re to be not less than the PSF size and not more than the Kron Radius\n',
                      '\n'])
        feedme_name = self.psf_double_sersic_folder + 'psf_double_sersic_' + str(self.obj_id) + '_' + str(self.NUMBER) + '.feedme'
        f = open(feedme_name,'w')
        f.writelines(['\n',
                      '===============================================================================\n',
                      '# IMAGE and GALFIT CONTROL PARAMETERS\n',
                      'A) ' + self.name_img_galfit + '      # Input data image (FITS file)\n',
                      'B) ' + outputname + '       # Output data image block\n',
                      'C) None # Sigma image name (made from data if blank or "none")\n',
                      'D) ' + self.name_psf_galfit + ' # Input PSF image and (optional) diffusion kernel\n',
                      'E) 1                   # PSF fine sampling factor relative to data\n',
                      'F) ' + self.name_mask_galfit + '  # Bad pixel mask (FITS image or ASCII coord list)\n',
                      'G) ' + constr_name + '   # File with parameter constraints (ASCII file) \n',
                      'H) 1    ' + str(self.imcut_dim) + ' 1  ' + str(self.imcut_dim) + '   # Image region to fit (xmin xmax ymin ymax)\n',
                      'I) ' + str(self.conv_box) + ' ' + str(self.conv_box) + ' # Size of the convolution box (x y)\n',
                      'J) ' + str(self.magzeropoint) + ' # Magnitude photometric zeropoint \n',
                      'K) ' + str(self.pixel_scale) + ' # Plate scale (dx dy)    [arcsec per pixel]\n',
                      'O) regular             # Display type (regular, curses, both)\n',
                      'P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n',
                      '\n',
                      '# INITIAL FITTING PARAMETERS\n',
                      '#\n',
                      '#   For object type, the allowed functions are: \n',
                      '#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n',
                      '#       ferrer, powsersic, sky, and isophote. \n',
                      '#\n',
                      '#   Hidden parameters will only appear when they are specified:\n',
                      '#       C0 (diskyness/boxyness), \n',
                      '#       Fn (n=integer, Azimuthal Fourier Modes),\n',
                      '#       R0-R10 (PA rotation, for creating spiral structures).\n',
                      '#\n',
                      '# -----------------------------------------------------------------------------\n',
                      '#   par)    par value(s)    fit toggle(s)    # parameter description\n',
                      '# -----------------------------------------------------------------------------\n',
                      '\n',
                      '# Object number: 1\n',
                      ' 0) psf                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '   1 1   #  position x, y\n',
                      ' 3) ' + str(self.mag_psf) + '     1          #  Integrated magnitude    \n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 2\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 4.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 3\n',
                      ' 0) sersic                 #  object type\n',
                      ' 1) ' + str(self.xy_position) + '  ' + str(self.xy_position) + '      1 1        #  position x, y\n',
                      ' 3) ' + str(self.mag_auto) + '     1          #  Integrated magnitude    \n',
                      ' 4) ' + str(self.radius) + '    1          #  R_e (half-light radius)   [pix]\n',
                      ' 5) 1.0000      1          #  Sersic index n (de Vaucouleurs n=4) \n',
                      ' 6) 0.0000      0          #     ----- \n',
                      ' 7) 0.0000      0          #     ----- \n',
                      ' 8) 0.0000      0          #     ----- \n',
                      ' 9) ' + str(self.ar) + '      1          #  axis ratio (b/a)  \n',
                      '10) ' + str(self.theta) + '    1          #  position angle (PA) [deg: Up=0, Left=90]\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract) \n',
                      '\n',
                      '# Object number: 4\n',
                      ' 0) sky                    #  object type\n',
                      ' 1) ' + str(self.sky) + '          1          #  sky background at center of fitting region [ADUs]\n',
                      ' 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n',
                      ' 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n',
                      ' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n',
                      '\n',
                      '================================================================================\n',
                      '\n'])
        
        f.close()