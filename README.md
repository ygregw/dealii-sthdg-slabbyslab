# Space-time HDG on moving domains (slab-by-slab)

This repository contains deal.II codes implementing:

- the space-time hybridizable discontinuous Galerkin method;
- for the advection-diffusion problem;
- on deforming domains;
- using the slab-by-slab approach.

## Demos

The mathematical description of the implemented test problem can be found in
**Section 6 - Numerical example** of [our
paper](https://arxiv.org/abs/2308.12130).

- A rotating Gaussian pulse:\
	<img src="https://github.com/ygregw/dealii-sthdg-slabbyslab/blob/main/misc/rot_pulse.gif" width="40%">
- Demo of the slab-by-slab approach:\
	<img src="https://github.com/ygregw/dealii-sthdg-slabbyslab/blob/main/misc/moving_slab.gif" width="40%">

# Setting up deal.II

## Installing PETSc

Useful links:

1. [Interfacing deal.II to PETSc](https://www.dealii.org/current/external-libs/petsc.html)
2. [PETSc's own installation instructions](https://petsc.org/release/install/download/#recommended-obtain-release-version-with-git)

```shell
$ cd /path/to/petsc
$ git clone -b release https://gitlab.com/petsc/petsc.git petsc-sthdg
$ cd petsc-sthdg
$ git checkout v3.17.1
$ git describe # you should see "v3.17.1"
$ export PETSC_DIR=`pwd`
$ export PETSC_ARCH=sthdg
$ ./configure --with-shared-libraries=1 --with-x=0 --with-mpi=1 --download-hypre=1
```

After successful configuration, apply the `makeflags.diff` patch inside the current `PETSC_DIR`:

```shell
$ basename $PWD # you should see "dealii-sthdg-slabbyslab"
$ cp ./misc/makeflags.diff $PETSC_DIR
$ cd $PETSC_DIR
$ git apply makeflags.diff
```

Now compile and check PETSc:

```shell
$ cd $PETSC_DIR
$ make PETSC_ARCH=sthdg all
$ # after successful compilation
$ make PETSC_ARCH=sthdg check
```

## Installing p4est

Useful links:

1. [Interfacing deal.II to p4est](https://www.dealii.org/developer/external-libs/p4est.html)
2. [p4est's own installation instructions](https://www.p4est.org/)

```shell
$ cd /path/to/p4est
$ wget https://github.com/p4est/p4est.github.io/blob/master/release/p4est-2.8.tar.gz
$ wget https://www.dealii.org/developer/external-libs/p4est-setup.sh
$ chmod u+x p4est-setup.sh
$ mkdir p4est-sthdg
$ ./p4est-setup.sh p4est-2.8.tar.gz `pwd`/p4est-sthdg
```

When successful, p4est is installed in `/path/to/p4est/p4est-sthdg`. This
directory is needed when compiling and installing deal.II. It might prove
convenient to assign it to an environment variable (better still, in your
`.bashrc`):

```shell
export P4EST_DIR=/path/to/p4est/p4est-sthdg
```

## Installing deal.II

Useful links:

1. [deal.II's own installation instructions](https://www.dealii.org/9.5.0/readme.html)
2. [Details on the deal.II configuration and build system](https://www.dealii.org/9.5.0/users/cmake_dealii.html)

```shell
$ cd /path/to/dealii
$ wget https://www.dealii.org/downloads/dealii-9.5.2.tar.gz
$ mkdir build install
$ tar xvzf dealii-9.5.2.tar.gz
$ mv dealii-9.5.2 source
$ cd build
$ cmake -DDEAL_II_WITH_PETSC=ON -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=sthdg -DDEAL_II_WITH_P4EST=ON -DP4EST_DIR=$P4EST_DIR -DDEAL_II_WITH_MPI=ON -DCMAKE_INSTALL_PREFIX=`pwd`/../install/ ../source
$ # after successful configuration
$ make --jobs=N install # specify number of jobs based on specs of your machine
```

# Running the space-time HDG code

Make sure you are in the root directory of this git repo. Then,
create `build` and `build/vtus` directories. The `vtus`
directory stores the VTK output files (to be visualized by
softwares like ParaView or VisIt).

```shell
$ basename $PWD # you should see "dealii-sthdg-slabbyslab"
$ mkdir -p build/vtus
```

Now configure and compile the code.

```shell
$ cd build
$ cmake ..
$ make
```

When successful, you should obtain executable `sthdg-advdif-slabbyslab`. It takes four commandline options:

1. `-o`: toggles vtu output on;
2. `-n N`: sets diffusion parameter to be 10^{-N};
3. `-c N`: sets N uniform refinement cycles;
4. `-p N`: uses finite elements of polynomial degree N.

Here is a test run with 4 mpi processes and its output:
```shell
$ mpiexec -n 4 ./sthdg-advdif-slabbyslab -n 8 -c 3 -p 1 | tee n8c3p1.txt
$ cat n8c3p1.txt
================================================================================
START DATE: 2024/5/28, TIME: 21:56:50
--------------------------------------------------------------------------------
Rotating Guassian Pulse Problem, nu = 1e-08
Running with 4 MPI processes, PETSC
Finite element space: FE_FaceQ<3>(1), FE_DGQ<3>(1)
Space-time IP-HDG, with semi-centered-flux penalty
================================================================================
--------------------------------------------------------------------------------
Cycle 1
--------------------------------------------------------------------------------
Set up system...
  Slab mesh: 	296 cells
  DoFHandler: 	4688 DoFs
  Mem usage: 	297 MB
  Time step: 	10 steps
Progress...
  25%...
  50%...
  75%...
  Done! (9.05645s)
Output results...
  Triple norm: 0.101185
--------------------------------------------------------------------------------
Cycle 2
--------------------------------------------------------------------------------
Set up system...
  Slab mesh: 	1100 cells
  DoFHandler: 	16800 DoFs
  Mem usage: 	312 MB
  Time step: 	20 steps
Progress...
  25%...
  50%...
  75%...
  Done! (63.5064s)
Output results...
  Triple norm: 0.0391919
--------------------------------------------------------------------------------
Cycle 3
--------------------------------------------------------------------------------
Set up system...
  Slab mesh: 	4372 cells
  DoFHandler: 	65216 DoFs
  Mem usage: 	330 MB
  Time step: 	40 steps
Progress...
  25%...
  50%...
  75%...
  Done! (516.759s)
Output results...
  Triple norm: 0.0113511
================================================================================
Convergence History:
--------------------------------------------------------------------------------
cells slabs dofs       L2          sH1          tH1         dif-jp       adv-jp        neum         supg        tnorm
  296    10  4688 2.0e-02    - 6.0e-05    - 1.2e-05    - 1.5e-05    - 3.6e-02    - 8.6e-02    - 3.5e-02    - 1.0e-01    -
 1100    20 16800 5.6e-03 1.95 3.0e-05 1.08 4.6e-06 1.46 8.9e-06 0.81 1.7e-02 1.17 3.3e-02 1.46 1.2e-02 1.60 3.9e-02 1.45
 4372    40 65216 1.1e-03 2.39 1.4e-05 1.06 1.6e-06 1.54 4.7e-06 0.93 6.5e-03 1.38 8.6e-03 1.94 3.4e-03 1.84 1.1e-02 1.80
================================================================================


+---------------------------------------------+------------+------------+
| Total wallclock time elapsed since start    |       591s |            |
|                                             |            |            |
| Section                         | no. calls |  wall time | % of total |
+---------------------------------+-----------+------------+------------+
| assemble_system                 |       146 |       303s |        51% |
| deform_slab                     |        73 |        70s |        12% |
| new_init_cond                   |        73 |      5.07s |      0.86% |
| setup_system                    |        73 |      50.2s |       8.5% |
| solve                           |        73 |       193s |        33% |
+---------------------------------+-----------+------------+------------+
