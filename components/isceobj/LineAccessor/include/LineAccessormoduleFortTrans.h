//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#ifndef LineAccessormoduleFortTrans_h
#define LineAccessormoduleFortTrans_h

        #if defined(NEEDS_F77_TRANSLATION)

                #if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)
                        #define LineAccessor_f lineaccessor_
                        #define getElements_f getelements_
                        #define getFileLength_f getfilelength_
                        #define getFileWidth_f getfilewidth_
                        #define getLineAccessorObject_f getlineaccessorobject_
                        #define getLineSequential_f getlinesequential_
                        #define getMachineEndianness_f getmachineendianness_
                        #define getSequentialElements_f getsequentialelements_
                        #define printAvailableDataTypesAndSizes_f printavailabledatatypesandsizes_
                        #define printObjectInfo_f printobjectinfo_
                        #define setChangeBandScheme_f setchangebandscheme_
                        #define setConvertFileEndianness_f setconvertfileendianness_
                        #define setElements_f setelements_
                        #define setFinalizeLineAccessor_f setfinalizelineaccessor_
                        #define setInitLineAccessor_f setinitlineaccessor_
                        #define setInitSequentialAccessor_f setinitsequentialaccessor_
                        #define setLineSequential_f setlinesequential_
                        #define setSequentialElements_f setsequentialelements_
                #elif defined(F77EXTERNS_NOTRAILINGBAR)
                        #define LineAccessor_f LineAccessor
                        #define getElements_f getElements
                        #define getFileLength_f getFileLength
                        #define getFileWidth_f getFileWidth
                        #define getLineAccessorObject_f getLineAccessorObject
                        #define getLineSequential_f getLineSequential
                        #define getMachineEndianness_f getMachineEndianness
                        #define getSequentialElements_f getSequentialElements
                        #define printAvailableDataTypesAndSizes_f printAvailableDataTypesAndSizes
                        #define printObjectInfo_f printObjectInfo
                        #define setChangeBandScheme_f setChangeBandScheme
                        #define setConvertFileEndianness_f setConvertFileEndianness
                        #define setElements_f setElements
                        #define setFinalizeLineAccessor_f setFinalizeLineAccessor
                        #define setInitLineAccessor_f setInitLineAccessor
                        #define setInitSequentialAccessor_f setInitSequentialAccessor
                        #define setLineSequential_f setLineSequential
                        #define setSequentialElements_f setSequentialElements
                #elif defined(F77EXTERNS_EXTRATRAILINGBAR)
                        #define LineAccessor_f LineAccessor__
                        #define getElements_f getElements__
                        #define getFileLength_f getFileLength__
                        #define getFileWidth_f getFileWidth__
                        #define getLineAccessorObject_f getLineAccessorObject__
                        #define getLineSequential_f getLineSequential__
                        #define getMachineEndianness_f getMachineEndianness__
                        #define getSequentialElements_f getSequentialElements__
                        #define printAvailableDataTypesAndSizes_f printAvailableDataTypesAndSizes__
                        #define printObjectInfo_f printObjectInfo__
                        #define setChangeBandScheme_f setChangeBandScheme__
                        #define setConvertFileEndianness_f setConvertFileEndianness__
                        #define setElements_f setElements__
                        #define setFinalizeLineAccessor_f setFinalizeLineAccessor__
                        #define setInitLineAccessor_f setInitLineAccessor__
                        #define setInitSequentialAccessor_f setInitSequentialAccessor__
                        #define setLineSequential_f setLineSequential__
                        #define setSequentialElements_f setSequentialElements__
                #elif defined(F77EXTERNS_UPPERCASE_NOTRAILINGBAR)
                        #define LineAccessor_f LINEACCESSOR
                        #define getElements_f GETELEMENTS
                        #define getFileLength_f GETFILELENGTH
                        #define getFileWidth_f GETFILEWIDTH
                        #define getLineAccessorObject_f GETLINEACCESSOROBJECT
                        #define getLineSequential_f GETLINESEQUENTIAL
                        #define getMachineEndianness_f GETMACHINEENDIANNESS
                        #define getSequentialElements_f GETSEQUENTIALELEMENTS
                        #define printAvailableDataTypesAndSizes_f PRINTAVAILABLEDATATYPESANDSIZES
                        #define printObjectInfo_f PRINTOBJECTINFO
                        #define setChangeBandScheme_f SETCHANGEBANDSCHEME
                        #define setConvertFileEndianness_f SETCONVERTFILEENDIANNESS
                        #define setElements_f SETELEMENTS
                        #define setFinalizeLineAccessor_f SETFINALIZELINEACCESSOR
                        #define setInitLineAccessor_f SETINITLINEACCESSOR
                        #define setInitSequentialAccessor_f SETINITSEQUENTIALACCESSOR
                        #define setLineSequential_f SETLINESEQUENTIAL
                        #define setSequentialElements_f SETSEQUENTIALELEMENTS
                #elif defined(F77EXTERNS_COMPAQ_F90)
                        #define LineAccessor_f LineAccessor_
                        #define getElements_f getElements_
                        #define getFileLength_f getFileLength_
                        #define getFileWidth_f getFileWidth_
                        #define getLineAccessorObject_f getLineAccessorObject_
                        #define getLineSequential_f getLineSequential_
                        #define getMachineEndianness_f getMachineEndianness_
                        #define getSequentialElements_f getSequentialElements_
                        #define printAvailableDataTypesAndSizes_f printAvailableDataTypesAndSizes_
                        #define printObjectInfo_f printObjectInfo_
                        #define setChangeBandScheme_f setChangeBandScheme_
                        #define setConvertFileEndianness_f setConvertFileEndianness_
                        #define setElements_f setElements_
                        #define setFinalizeLineAccessor_f setFinalizeLineAccessor_
                        #define setInitLineAccessor_f setInitLineAccessor_
                        #define setInitSequentialAccessor_f setInitSequentialAccessor_
                        #define setLineSequential_f setLineSequential_
                        #define setSequentialElements_f setSequentialElements_
                #else
                        #error Unknown traslation for FORTRAN external symbols
                #endif

        #endif

#endif //LineAccessormoduleFortTrans_h
