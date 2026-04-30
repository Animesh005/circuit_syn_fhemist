# Circuit synthesis using FHEMIST framework

## Enable Heterogeneous Managed Memory (HMM) from NVIDIA

For enabling and detecting HMM in the system, follow the steps from: https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/

## Run only CMIS optimization

```bash
python3 main.py
```
Enter the path for input netlist: circuits/EPFL_circuits/adder.txt

## Run the hybrid CMIS+CSO optimization

```bash
python3 cso_synthesis.py
```
Enter the path for input netlist: circuits/EPFL_circuits/adder.txt
Enter the path for output netlist: circuits/CSO_circuits/adder.txt

This writes the CSO optimized circuits in the circuits/CSO_circuits location.

### Now, run the CMIS optimization over the CSO optimized circuits.

```bash
python3 main.py
```
Enter the path for input netlist: circuits/CSO_circuits/adder.txt
