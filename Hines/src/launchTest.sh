#!/bin/bash

rm results0.dat -f
rm results1.dat -f

rm testDir -rf 
mkdir testDir

#
echo "1) Building non-MPI version"
rm *.o HinesGpu -f
./buildLinux.sh > testDir/buildLinux.dat 2>&1

#
echo "- Running version with 1 thread [C|G|H]"
./HinesGpu C 1000 4 1 n1l 1 > testDir/C1.dat 2>&1
./HinesGpu G 1000 4 1 n1l 1 > testDir/G1.dat 2>&1
./HinesGpu H 1000 4 1 n1l 1 > testDir/H1.dat 2>&1
cat testDir/C1.dat | grep meanGenSpikes
cat testDir/G1.dat | grep meanGenSpikes
cat testDir/H1.dat | grep meanGenSpikes

echo "- Running version with 2 threads [C|G|H]"
./HinesGpu C 2000 4 2 n1l 1 > testDir/C2.dat 2>&1
./HinesGpu G 2000 4 2 n1l 1 > testDir/G2.dat 2>&1
./HinesGpu H 2000 4 2 n1l 1 > testDir/H2.dat 2>&1
cat testDir/C2.dat | grep meanGenSpikes
cat testDir/G2.dat | grep meanGenSpikes
cat testDir/H2.dat | grep meanGenSpikes

#
echo
echo "2) Building MPI version"
rm *.o HinesGpu -f
./buildLinuxMpi.sh > testDir/buildLinuxMpi.dat 2>&1

#
echo "- Running version with 1 process 1 thread [C|G|H]"
mpirun -np 1 ./HinesGpu C 1000 4 1 n1l 1 > testDir/C1m1.dat 2>&1
mpirun -np 1 ./HinesGpu G 1000 4 1 n1l 1 > testDir/G1m1.dat 2>&1
mpirun -np 1 ./HinesGpu H 1000 4 1 n1l 1 > testDir/H1m1.dat 2>&1
cat testDir/C1m1.dat | grep meanGenSpikes
cat testDir/G1m1.dat | grep meanGenSpikes
cat testDir/H1m1.dat | grep meanGenSpikes

echo "- Running version with 2 process 1 threads [C|G|H]"
mpirun -np 2 ./HinesGpu C 2000 4 1 n1l 1 > testDir/C1m2.dat 2>&1
mpirun -np 2 ./HinesGpu G 2000 4 1 n1l 1 > testDir/G1m2.dat 2>&1
mpirun -np 2 ./HinesGpu H 2000 4 1 n1l 1 > testDir/H1m2.dat 2>&1
cat testDir/C1m2.dat | grep meanGenSpikes
cat testDir/G1m2.dat | grep meanGenSpikes
cat testDir/H1m2.dat | grep meanGenSpikes

echo "- Running version with 2 processes 2 threads [C|G|H]"
mpirun -np 2 ./HinesGpu C 2000 4 2 n1l 1 > testDir/C2m2.dat 2>&1
mpirun -np 2 ./HinesGpu G 2000 4 2 n1l 1 > testDir/G2m2.dat 2>&1
mpirun -np 2 ./HinesGpu H 2000 4 2 n1l 1 > testDir/H2m2.dat 2>&1
cat testDir/C2m2.dat | grep meanGenSpikes
cat testDir/G2m2.dat | grep meanGenSpikes
cat testDir/H2m2.dat | grep meanGenSpikes



#A="4 8 12 16 20 24 28 32"
#for i in $A
#do
#   echo "------------------------------------------------------"
#   ./HinesGpu1 G 10000 $i
#   ./HinesGpu3 G 10000 $i 1
#done
