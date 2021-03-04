export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7

#scansar-scansar
##########################
cd scansar-scansar/1
alos2App.py --steps
cd ../../

cd scansar-scansar/2
alos2App.py --steps
cd ../../

cd scansar-scansar/3
alos2App.py --steps
cd ../../

cd scansar-scansar/4
alos2App.py --steps
cd ../../


#scansar-stripmap
##########################
cd scansar-stripmap/1
alos2App.py --steps
cd ../../

cd scansar-stripmap/2
alos2App.py --steps
cd ../../


#stripmap-stripmap
##########################
cd stripmap-stripmap/1
alos2App.py --steps
cd ../../

cd stripmap-stripmap/2
alos2App.py --steps
cd ../../

cd stripmap-stripmap/3
alos2App.py --steps
cd ../../

cd stripmap-stripmap/4
alos2App.py --steps
cd ../../


#scansar-scansar_7s
##########################
cd scansar-scansar_7s
alos2App.py --steps
cd ../
