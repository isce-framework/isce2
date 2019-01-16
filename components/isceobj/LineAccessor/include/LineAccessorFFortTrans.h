
#ifndef LineAccessorFFortTrans_h
#define LineAccessorFFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define getMachineEndianness_f getmachineendianness_
                        #define getLineAccessorObject_f getlineaccessorobject_
                        #define changeBandScheme_f changebandscheme_
                        #define convertFileEndianness_f convertfileendianness_
                        #define finalizeLineAccessor_f finalizelineaccessor_
                        #define initLineAccessor_f initlineaccessor_
                        #define createFile_f createfile_
                        #define rewindImage_f rewindimage_
                        #define printArray_f printarray_
                        #define printAvailableDataTypesAndSizes_f printavailabledatatypesandsizes_
                        #define printObjectInfo_f printobjectinfo_
                        #define getFileWidth_f getfilewidth_
                        #define getTypeSize_f gettypesize_
                        #define getFileLength_f getfilelength_
                        #define setLineSequential_f setlinesequential_
                        #define setLine_f setline_
                        #define setStream_f setstream_
                        #define setStreamAtPos_f setstreamatpos_
                        #define getStream_f getstream_
                        #define getStreamAtPos_f getstreamatpos_
                        #define setSequentialElements_f setsequentialelements_
                        #define getLineSequential_f getlinesequential_
                        #define getLine_f getline_
                        #define getSequentialElements_f getsequentialelements_
                        #define getElements_f getelements_
                        #define setElements_f setelements_
                        #define initSequentialAccessor_f initsequentialaccessor_
                #else
                        #error Unknown traslation for FORTRAN external symbols
                #endif

        #endif

#endif //LineAccessorFFortTrans_h
