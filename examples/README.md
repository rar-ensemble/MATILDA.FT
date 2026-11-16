This folder contains several read-to-go demonstration simulations that highlight some of the features of MATILDA.FT.

PARTICLE-BASED MODELS:

From the subfolder containing the input file, run the command:

coacervate:
Particle-based simulation of the system considered in Riggleman, Kumar, and Fredrickson (J. Chem. Phys. 2012). This particular simulation is run at dimensionless excluded volume parameter B = 0.05 and dimensionless Bjerrum length E = 10000
.

dpd:
Runs a spinodal demixing simulation using the DPD thermostat.


lamella:
A 3D simulation of an A-B diblock copolymer with f = 0.5 and large chi*N. This system is randomly initialized and is expected to form a defective lamellar state.


model-A:
Particle-based implementation of the Gaussian-regularized Edwards model by Villet and Fredrickson (J. Chem. Phys. 2014). 


liquid-crystal:
Small liquid crystal system that ramps the Maier-Saupe parameter from 5 to 120. The mesogens are discretized into three interaction sites with the Maier-Saupe potential acting on the center. This simulation should exhibit both an isotropic-nematic transition and a nematic-smecticA transition.



FIELD-THEORETIC SIMULATIONS

From the subfolder containing the input file, run the command:

../../matilda.ft -ft

ft:
Calculates the equilibrium structure of an A-B homopolymer blend at chi*N = 3.5. The system is initialized to a sine wave potential profile.

lc-scft:
2D SCFT of a three-site rigid mesogen with a Maier-Saupe interaction on the central site. Seeded with a density wave, it converges to a smectic-A phase with a layer spacing of three bond lengths. Run with ../../matilda.ft -in input

lc-scft-3d:
3D SCFT of a three-site rigid mesogen with a Maier-Saupe interaction on every site, using a 110-point Lebedev rule for the orientations. Converges to a uniform nematic with order parameter 0.747. Run with ../../matilda.ft -in input

lc-interface-scft:
2D SCFT of a three-site rigid mesogen with sites A-A-B blended with a C homopolymer. A and C are strongly incompatible while the B end is only weakly repelled by C, so the mesogens and polymer phase separate and the B ends are enriched on the polymer side of the interface. Run with ../../matilda.ft -in input

