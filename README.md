# Space-time HDG on moving domains (slab-by-slab)

This repository contains deal.II codes implementing:

- the space-time hybridizable discontinuous Galerkin method;
- for the advection-diffusion problem;
- on deforming domains;
- using the slab-by-slab approach.

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

## Installing PETSc

1. [deal.II's own installation instructions](https://www.dealii.org/9.5.0/readme.html)

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
