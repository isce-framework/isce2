#!/usr/bin/env python3

import os

Import('envcontrib')

envPyCuAmpcor = envcontrib.Clone()
package = envPyCuAmpcor['PACKAGE']
project = 'PyCuAmpcor'
envPyCuAmpcor['PROJECT'] = project

Export('envPyCuAmpcor')

if envPyCuAmpcor['GPU_ACC_ENABLED']:
    envPyCuAmpcor.Append(CPPPATH=envPyCuAmpcor['CUDACPPPATH'])
    envPyCuAmpcor.Append(LIBPATH=envPyCuAmpcor['CUDALIBPATH'])
    envPyCuAmpcor.Append(LIBS=['cuda','cudart','cufft','cublas'])
    build = envPyCuAmpcor['PRJ_SCONS_BUILD'] + '/' + package + '/' + project

#    includeScons = os.path.join('include','SConscript')
#    SConscript(includeScons)

    cudaScons = os.path.join('src', 'SConscript')
    SConscript(cudaScons, variant_dir=os.path.join(envPyCuAmpcor['PRJ_SCONS_BUILD'], package, project, 'src'))

    install = os.path.join(envPyCuAmpcor['PRJ_SCONS_INSTALL'],package,project)
    initFile = '__init__.py'

    if not os.path.exists(initFile):
        with open(initFile, 'w') as fout:
            fout.write("#!/usr/bin/env python3")

    listFiles = [initFile]
    envPyCuAmpcor.Install(install, listFiles)
    envPyCuAmpcor.Alias('install', install)
