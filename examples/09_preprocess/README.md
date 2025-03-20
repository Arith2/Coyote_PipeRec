# Coyote Example 9: DLRM Preprocessing Tasks 
# (In Progress)
This example is used to demonstrate how to integrate FPGA for prerpocessing tasks and GPU for DLRM training.

1. hw: this folder contains files to do preprocessing tasks in FPGA
2. sw: this folder contains files to initiate data movement from the CPU side. Later we would manage to combine it with the P2P example.

In the current stage, the pipeline contains several stateless operators.
It reads data from CPU memory and transfer data back to CPU.

