from iscesys.Component.Component import Component


#This one parameter also appears in InsarProc.py to tell the code not to handle
#this parameter in the case when the user does not give information.  The
#mandatory=False, private=True case is for a truly optional case in which the
#code is happy not to have a value for the parameter.
NUMBER_VALID_PULSES = Component.Parameter('_numberValidPulses',
                                          public_name='numberValidPulses',
                                          default=2048,
                                          type=int,
                                          mandatory=False,
                                          private=True,
                                          doc='')

#The rest of these parameters are mandatory=True, private=True and are hidden
#from the user because the (True, True) state is meant to communicate to the
#code that these parameters must be set before execution of code.
PEG_H1 = Component.Parameter('_pegH1',
                             public_name='pegH1',
                             default=None,
                             type=float,
                             mandatory=True,
                             private=True,
                             doc='')


PEG_H2 = Component.Parameter('_pegH2',
                             public_name='pegH2',
                             default=None,
                             type=float,
                             mandatory=True,
                             private=True,
                             doc='')


FD_H1 = Component.Parameter('_fdH1',
                            public_name='fdH1',
                            default=None,
                            type=float,
                            mandatory=True,
                            private=True,
                            doc='')


FD_H2 = Component.Parameter('_fdH2',
                            public_name='fdH2',
                            default=None,
                            type=float,
                            mandatory=True,
                            private=True,
                            doc='')


PEG_V1 = Component.Parameter('_pegV1',
                             public_name='pegV1',
                             default=None,
                             type=float,
                             mandatory=True,
                             private=True,
                             doc='')

PEG_V2 = Component.Parameter('_pegV2',
                             public_name='pegV2',
                             default=None,
                             type=float,
                             mandatory=True,
                             private=True,
                             doc='')

#ask
NUMBER_RANGE_BINS = Component.Parameter('_numberRangeBins',
                                        public_name='numberRangeBins',
                                        default=None,
                                        type=int,
                                        mandatory=True,
                                        private=True,
                                        doc='')


MACHINE_ENDIANNESS = Component.Parameter('_machineEndianness',
                                         public_name='machineEndianness',
                                         default='l',
                                         type=str,
                                         mandatory=True,
                                         private=True,
                                         doc='')
#ask
CHIRP_EXTENSION = Component.Parameter('_chirpExtension',
                                      public_name='chirpExtension',
                                      default=0,
                                      type=int,
                                      mandatory=True,
                                      private=True,
                                      doc='')
#ask
SLANT_RANGE_PIXEL_SPACING = Component.Parameter('_slantRangePixelSpacing',
                                                public_name='slantRangePixelSpacing',
                                                default=None,
                                                type=float,
                                                mandatory=True,
                                                private=True,
                                                doc='')
#ask
NUMBER_RESAMP_LINES = Component.Parameter('_numberResampLines',
                                          public_name='numberResampLines',
                                          default=None,
                                          type=int,
                                          mandatory=True,
                                          private=True,
                                          doc='')
LOOK_SIDE = Component.Parameter('_lookSide',
                                public_name='lookSide',
                                default=-1,
                                type=int,
                                mandatory=True,
                                private=True,
                                doc='')

REFERENCE_FRAME = Component.Facility('_referenceFrame',
                                   public_name='referenceFrame',
                                   factory='default',
                                   mandatory=True,
                                   private=True,
                                   doc='Reference frame')


SECONDARY_FRAME = Component.Facility('_secondaryFrame',
                                  public_name='secondaryFrame',
                                  factory='default',
                                  mandatory=True,
                                  private=True,
                                  doc='Secondary frame')


REFERENCE_ORBIT = Component.Facility('_referenceOrbit',
                                   public_name='referenceOrbit',
                                   factory='default',
                                   mandatory=True,
                                   private=True,
                                   doc='Reference orbit')


SECONDARY_ORBIT = Component.Facility('_secondaryOrbit',
                                  public_name='secondaryOrbit',
                                  factory='default',
                                  mandatory=True,
                                  private=True,
                                  doc='Secondary orbit')

#ask
DOPPLER_CENTROID = Component.Facility('_dopplerCentroid',
                                       public_name='dopplerCentroid',
                                       factory='default',
                                       mandatory=True,
                                       private=True,
                                       doc='')

REFERENCE_DOPPLER = Component.Facility('_referenceDoppler',
                                     public_name='referenceDoppler',
                                     factory='default',
                                     mandatory=True,
                                     private=True,
                                     doc='')


SECONDARY_DOPPLER = Component.Facility('_secondaryDoppler',
                                    public_name='secondaryDoppler',
                                    factory='default',
                                    mandatory=True,
                                    private=True,
                                    doc='')

REFERENCE_RAW_IMAGE = Component.Facility('_referenceRawImage',
                                       public_name='referenceRawImage',
                                       factory='default',
                                       mandatory=True,
                                       private=True,
                                       doc='')


SECONDARY_RAW_IMAGE = Component.Facility('_secondaryRawImage',
                                      public_name='secondaryRawImage',
                                      factory='default',
                                      mandatory=True,
                                      private=True,
                                      doc='')


REFERENCE_SLC_IMAGE = Component.Facility('_referenceSlcImage',
                                       public_name='referenceSlcImage',
                                       factory='default',
                                       mandatory=True,
                                       private=True,
                                       doc='')


SECONDARY_SLC_IMAGE = Component.Facility('_secondarySlcImage',
                                      public_name='secondarySlcImage',
                                      factory='default',
                                      mandatory=True,
                                      private=True,
                                      doc='')


OFFSET_AZIMUTH_IMAGE = Component.Facility('_offsetAzimuthImage',
                                           public_name='offsetAzimuthImage',
                                           factory='default',
                                           mandatory=True,
                                           private=True,
                                           doc='')


OFFSET_RANGE_IMAGE = Component.Facility('_offsetRangeImage',
                                         public_name='offsetRangeImage',
                                         factory='default',
                                         mandatory=True,
                                         private=True,
                                         doc='')


RESAMP_AMP_IMAGE = Component.Facility('_resampAmpImage',
                                       public_name='resampAmpImage',
                                       factory='default',
                                       mandatory=True,
                                       private=True,
                                       doc='')


RESAMP_INT_IMAGE = Component.Facility('_resampIntImage',
                                       public_name='resampIntImage',
                                       factory='default',
                                       mandatory=True,
                                       private=True,
                                       doc='')


RESAMP_ONLY_IMAGE = Component.Facility('_resampOnlyImage',
                                        public_name='resampOnlyImage',
                                        factory='default',
                                        mandatory=True,
                                        private=True,
                                        doc='')


RESAMP_ONLY_AMP = Component.Facility('_resampOnlyAmp',
                                      public_name='resampOnlyAmp',
                                      factory='default',
                                      mandatory=True,
                                      private=True,
                                      doc='')


TOPO_INT_IMAGE = Component.Facility('_topoIntImage',
                                     public_name='topoIntImage',
                                     factory='default',
                                     mandatory=True,
                                     private=True,
                                     doc='')


HEIGHT_TOPO_IMAGE = Component.Facility('_heightTopoImage',
                                        public_name='heightTopoImage',
                                        factory='default',
                                        mandatory=True,
                                        private=True,
                                        doc='')

RG_IMAGE = Component.Facility('_rgImage',
                               public_name='rgImage',
                               factory='default',
                               mandatory=True,
                               private=True,
                               doc='')

SIM_AMP_IMAGE = Component.Facility('_simAmpImage',
                                    public_name='simAmpImage',
                                    factory='default',
                                    mandatory=True,
                                    private=True,
                                    doc='')

WBD_IMAGE = Component.Facility('_wbdImage',
                                    public_name='wbdImage',
                                    factory='default',
                                    mandatory=True,
                                    private=True,
                                    doc='')

DEM_IMAGE = Component.Facility('_demImage',
                                public_name='demImage',
                                factory='default',
                                mandatory=True,
                                private=True,
                                doc='')


FORM_SLC1 = Component.Facility('_formSLC1',
                                  public_name='formSLC1',
                                  factory='default',
                                  mandatory=True,
                                  private=True,
                                  doc='')


FORM_SLC2 = Component.Facility('_formSLC2',
                                  public_name='formSLC2',
                                  factory='default',
                                  mandatory=True,
                                  private=True,
                                  doc='')


MOCOMP_BASELINE = Component.Facility('_mocompBaseline',
                                      public_name='mocompBaseline',
                                      factory='default',
                                      mandatory=True,
                                      private=True,
                                      doc='')


TOPOCORRECT = Component.Facility('_topocorrect',
                                  public_name='topocorrect',
                                  factory='default',
                                  mandatory=True,
                                  private=True,
                                  doc='')


TOPO = Component.Facility('_topo',
                           public_name='topo',
                           factory='default',
                           mandatory=True,
                           private=True,
                           doc='')

RAW_REFERENCE_IQ_IMAGE = Component.Facility('_rawReferenceIQImage',
                                           public_name='rawReferenceIQImage',
                                           factory='default',
                                           mandatory=True,
                                           private=True,
                                           doc='')


RAW_SECONDARY_IQ_IMAGE = Component.Facility('_rawSecondaryIQImage',
                                          public_name='rawSecondaryIQImage',
                                          factory='default',
                                          mandatory=True,
                                          private=True,
                                          doc='')
TOPOCORRECT_FLAT_IMAGE = Component.Facility('_topocorrectFlatImage',
                                             public_name='topocorrectFlatImage',
                                             factory='default',
                                             mandatory=True,
                                             private=True,
                                             doc='')


#i know the answer but double check
OFFSET_FIELD = Component.Facility('_offsetField',
                                   public_name='offsetField',
                                   factory='default',
                                   mandatory=True,
                                   private=True,
                                   doc='')


REFINED_OFFSET_FIELD = Component.Facility('_refinedOffsetField',
                                           public_name='refinedOffsetField',
                                           factory='default',
                                           mandatory=True,
                                           private=True,
                                           doc='')


OFFSET_FIELD1 = Component.Facility('_offsetField1',
                                    public_name='offsetField1',
                                    factory='default',
                                    mandatory=True,
                                    private=True,
                                    doc='')


REFINED_OFFSET_FIELD1 = Component.Facility('_refinedOffsetField1',
                                            public_name='refinedOffsetField1',
                                            factory='default',
                                            mandatory=True,
                                            private=True,
                                            doc='')

parameter_list = (
                      PEG_H1,
                      PEG_H2,
                      FD_H1,
                      FD_H2,
                      PEG_V1,
                      PEG_V2,
                      NUMBER_RANGE_BINS,
                      MACHINE_ENDIANNESS,
                      CHIRP_EXTENSION,
                      SLANT_RANGE_PIXEL_SPACING,
                      LOOK_SIDE,
                      NUMBER_RESAMP_LINES
                     )
facility_list = (
                    REFERENCE_FRAME,
                    SECONDARY_FRAME,
                    REFERENCE_ORBIT,
                    SECONDARY_ORBIT,
                    REFERENCE_DOPPLER,
                    SECONDARY_DOPPLER,
                    DOPPLER_CENTROID,
                    REFERENCE_RAW_IMAGE,
                    SECONDARY_RAW_IMAGE,
                    REFERENCE_SLC_IMAGE,
                    SECONDARY_SLC_IMAGE,
                    OFFSET_AZIMUTH_IMAGE,
                    OFFSET_RANGE_IMAGE,
                    RESAMP_AMP_IMAGE,
                    RESAMP_INT_IMAGE,
                    RESAMP_ONLY_IMAGE,
                    RESAMP_ONLY_AMP,
                    TOPO_INT_IMAGE,
                    HEIGHT_TOPO_IMAGE,
                    RG_IMAGE,
                    SIM_AMP_IMAGE,
                    DEM_IMAGE,
                    FORM_SLC1,
                    FORM_SLC2,
                    MOCOMP_BASELINE,
                    TOPOCORRECT,
                    TOPO,
                    RAW_REFERENCE_IQ_IMAGE,
                    RAW_SECONDARY_IQ_IMAGE,
                    TOPOCORRECT_FLAT_IMAGE,
                    OFFSET_FIELD,
                    REFINED_OFFSET_FIELD,
                    OFFSET_FIELD1,
                    REFINED_OFFSET_FIELD1,
                    WBD_IMAGE
                 )
