# MiniAMR-DPCPP
A port of the Mantevo MiniAMR mini-app using DPCPP.

Description of folders in no particular order

For original see: https://github.com/Mantevo/miniAMR \
Overall execution sudo-code from: \
C. T. Vaughan and R. F. Barrett, "Enabling Tractable Exploration of the Performance of Adaptive Mesh Refinement," 2015 IEEE International Conference on Cluster Computing, Chicago, IL, 2015, pp. 746-752, doi: 10.1109/CLUSTER.2015.129.

### dpcpp

First half decent dpcpp port of the miniAMR app. This was after realizing some major flaws in prior executions and this is what all other tests draw their lineage from.

Overall Execution:

```
for some number of timesteps do
    for some number of stages do
        communicate ghost values between blocks
        for some number of variables
            perform stencil calculation on variables
            if stage for checksums then
                perform checksum calculations
                compare checksum values
            end if
        end for
    end for
    if time for refinement then
        refine mesh
    end if
end for
```

Stencil Execution:

for some number of blocks
    flatten bp->array from 4d to 1d
    store in inputArray
    create blank array outputArray
    create buffer using inputArray
    create buffer using outputArray
    send buffers to fpga kernel
    store outputArray into bp->array
end for

Kernel Execution:

for all cells sent to kernel
    store into local_array BRAM
end for
for all cells sent to kernel
    calculate 7 point stencil
    store into work
end for
for all cells sent to kernel
    store work into outputArray
end for

### flattened

Based on: memorycombine

In this iteration of the code all references to bp->array[var][i][j][k] were flattened into a 1d array. This allows buffers to be created directly with the bp->array pointer which reduces the amount of memory movements.

Overall Execution:

for some number of timesteps do
    for some number of stages do
        communicate ghost values between blocks
        perform stencil calculation on variables
        if stage for checksums then
            for some number of variables
                perform checksum calculations
                compare checksum values
            end for
        end if
    end for
    if time for refinement then
        refine mesh
    end if
end for

Stencil Execution:

for some number of blocks
    create buffer using bp->array
    send buffer to fpga kernel
end for

Kernel Execution:

for some number of variables
    for all cells sent to kernel
        store inputArray into local_array BRAM
    end for
    for all stored cells
        calculate 7 point stencil
        store into inputArray
    end for
end for
  
