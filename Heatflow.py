# Import relevant packages
#import discretize
#from discretize import TensorMesh
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcol
import matplotlib.ticker as ticker
from scipy.constants import R as R
from scipy.constants import N_A as N_A

# GLOBAL VARIABLES
# Materials
class gas():
    def __init__(self):
        self.density = 1.15
        self.conductivity = 0.024
        self.viscosity = 1.7e-5
        self.capacity = 720
        self.velocity = 50e-3 / (np.pi * (27*0.001)**2)
gas = gas()

class steel():
    def __init__(self):
        self.density = 8030
        self.conductivity = 16.3

    def capacity(self, T):
        return (450 + 0.28 * (T - 273))
    
steel = steel()

class cordierite():
    def __init__(self):
        self.density = 2300
        self.conductivity = 2.5
        self.capacity = 900
cordierite = cordierite()

solidGasBoundaryC = 2
cordieriteSteelBoundaryC = np.infty

# Temperatures
Tair = 25 + 273
T0 = 25 + 273
Tgas = 300 + 273
Tcatalyst = 150 + 273

# Dimensions (all in mm)
class pipe():
    def __init__(self):
        self.thickness = 1

        self.outerRadius = 28 #56 * 0.5
        self.innerRadius = self.outerRadius - self.thickness
        self.length = 100
pipe = pipe()

class air():
    def __init__(self):
        self.conductivity = 0.024
        self.capacity = 720
        self.density = 1
air = air()

class substrate():
    def __init__(self):
        self.length = 10
        self.thickness = 1
        self.channel = 4
substrate = substrate()

# Discretization setup
elementDiscretisation = 200
dz = 1 # volume elements of 1 mm
dx = dz # volume elements are cubes (also dy = dx)
dy = dx
dt = elementDiscretisation * (dz * 0.001 / gas.velocity)
faceArea = dx ** 2 * 1e-6 # area of a face of a volume element
faceL = dx / 2 # distance between the centre of a volume element and its face
volumeOfElement = dx ** 3 * 1e-9

# Space
zspace = int(pipe.length / dz) # no. of elements in longitudinal direction
xspace = int(pipe.outerRadius / dx) # no. of elements in radial direction from centre
yspace = xspace


# Dimensions in element size
elementLengthCordierite = int((substrate.length / pipe.length) * (zspace + 1))
velocityRatio = 1.5
elementsMoved = [elementDiscretisation, velocityRatio * elementDiscretisation]
# velocity in element length [not in cordierite, in cordierite]

# Plot setup
downstreamSurfaceInterval = 100

# TensorMesh gris setup
x = np.linspace(0, xspace + 1, xspace + 2) - 0.5
y = np.linspace(0, yspace + 1, yspace + 2) - 0.5
z = np.linspace(0, zspace, zspace + 1 )- 0.5

cx = np.linspace(0, xspace - 1, xspace) + 0.5
cy = np.linspace(0, yspace - 1, yspace) + 0.5
cz = np.linspace(0, zspace - 1 , zspace) + 0.5
X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')


def channelVolumeElements():
    """
    CHANNELVOLUMEELEMENTS returns the amount of volume elements inside
    all of the channels inside the cross-section of the cordierite
    substrate (at a single z-coordinate). Note that this also includes
    the volume elements that hit the downstream surface of the
    cordierite, hence it is effectively the number of volume elements
    inside the pipe within a cross-section.
    """
    xy = materialSpacexyz()[1, 1:-1, 1:-1] # takes one xy plane
    XX = X[1:-1, 1:-1, zspace - 1]
    YY = Y[1:-1, 1:-1,zspace - 1]
    channelVolumeElements = 0
    for ix in range(xspace): # iterate over all volume elements
        for iy in range(yspace):
            r2 = XX[xspace - 1 - ix, iy] ** 2 + YY[xspace - 1 - ix, iy] ** 2
            if r2 <= (pipe.innerRadius ** 2) and xy[ix, iy] == 0:
                channelVolumeElements += 1
            else:
                continue
    return channelVolumeElements

def velocityChange():
    """
    VELOCITYCHANGE calculates the change of velocity (given as a fraction) of
    the exhaust gas once it enters the cordierite matrix, as volume flow rate
    is constant, v1 * A1 = v2 * A2, where v and A is velocity and area
    respectively.
    """
    materialSpace = materialSpacexyz()
    xy = materialSpace[zspace - 1, 1:-1, 1:-1] # takes one xy plane
    Acordierite = np.count_nonzero(xy == 3) # cordierite area
    Achannel = channelVolumeElements() # channel area (empty space filled with gas/air)
    return (Acordierite + Achannel) / (Achannel)

def materialSpacexyz():
    """
    MATERIALSPACE creates a 3D numpy array of the material space. Entering bo
    undary
    is set to be the gas flowing onto the pipe and the entire exit boundary is set
    to be an air reservoir. The positive z-direction goes along the longitudinal
    length of the pipe. whilst the positive x and y-directions go from the radius
    inward, towards the centre of the pipe's cross-section.
    0 = air
    1 = gas
    2 = steel
    3 = cordierite
    """
    materialSpace = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # initialise system
    for ix in range(xspace + 2):
        for iy in range(yspace + 2):
            for iz in range(1, zspace + 1): # while in the pipe length

                r2 = X[xspace + 1 - ix, iy, iz] ** 2 + Y[xspace + 1 - ix, iy, iz] ** 2
                if r2 > pipe.outerRadius ** 2:
                    materialSpace[iz, ix, iy] = 0 # overlay air
                elif r2 > pipe.innerRadius ** 2 and r2 < pipe.outerRadius ** 2:
                    materialSpace[iz, ix, iy] = 2 # overlay pipe
                elif r2 < pipe.outerRadius ** 2:
                    if Z[xspace + 1 - ix, iy, iz] > zspace - substrate.length:
                        if (X[xspace + 1 - ix, iy, iz] - 2.5) % 5 == 0:
                            materialSpace[iz, ix, iy] = 3 # horizontal cordierite
                        if (Y[ix, iy, iz] - 2.5) % 5 == 0:
                            materialSpace[iz, ix, iy] = 3 # vertical cordierite
    return materialSpace

 # Adsorption definitions
Constant = 4e27
activationEnergy = 30e3
dA = dx ** 2 * 1e-6

exitGasV = channelVolumeElements() * elementsMoved[1] * 1e-9 # volume of exiting gas in m^3
molGas = exitGasV / 24e-3 # molar gas volume
nCO_end = molGas * 0.001 * N_A # final nCO threshold
nCO_fresh = molGas * 0.01 * N_A # starting nCO

# Thermal equilibrium setup
deltaTeq = 0.05
equilibriumTime = np.nan

def gasFlow(tempProfile, materialSpace):
    """
    GASFLOW moves the gas volume elements by distance ds moved in time dt.
    The distance moved ds is defined to be two volume elements outside of
    cordierite and three volume elements inside cordierite based on the
    velocity change from VELOCITYCHANGE().
    """
    newTempProfile = np.copy(tempProfile)
    newMaterialProfile = np.copy(materialSpace)
    for ix in range(xspace + 2): # iterate over all volume elements
        for iy in range(yspace + 2):
            for iz in range(1, zspace + 1):
                r2 = X[xspace + 1 - ix, iy, iz] ** 2 + Y[xspace + 1 - ix, iy, iz] ** 2
                air = materialSpace[iz, ix, iy] == 0
                gas = materialSpace[iz, ix, iy] == 1
                air_gas = air or gas
                if r2 < pipe.innerRadius ** 2 and air_gas: # find air within the pipe,
                    if iz < zspace - elementLengthCordierite + 1: # outside cordierite
                        v = elementsMoved[0]
                    else: # inside cordierite
                        v = elementsMoved[1]
                    if iz - v >= 0: # assign moving temperature and material type
                        newTempProfile[iz, ix, iy] = tempProfile[iz - v, ix, iy]
                        #newTempProfile[iz, ix, iy] = Tgas
                        newMaterialProfile[iz, ix, iy] = materialSpace[iz - v, ix, iy]

                    else: # assign temperature and material type of incoming gas
                        newTempProfile[iz, ix, iy] = tempProfile[0, ix, iy]
                        newTempProfile[iz, ix, iy] = Tgas
                        #newMaterialProfile[iz, ix, iy] = materialSpace[0, ix, iy]
                else:
                    continue
    return newTempProfile, newMaterialProfile

def propertySpace(tempProfile, materialSpace):
    """
    PROPERTYSPACE creates 3D numpy arrays made of the conductivity, density a
    nd
    specific heat capacity of each volume element's material. The coordinate
    s
    (position) of these values in these spaces then correspond to the position
    of the volume elements in MATERIALSPACE.
    """
    # Initialise the arrays (spaces)
    densitySpace = np.copy(materialSpace)
    kSpace = np.copy(materialSpace)
    capacitySpace = np.copy(materialSpace)

    # Assign densities to each volume element's cordinate
    densitySpace = np.where(densitySpace == 0, air.density,densitySpace)
    densitySpace = np.where(densitySpace == 1, gas.density,densitySpace)
    densitySpace = np.where(densitySpace == 2, steel.density,densitySpace)
    densitySpace = np.where(densitySpace == 3, cordierite.density,densitySpace)

    # Assign conductivities to each volume element's cordinate
    kSpace = np.where(kSpace == 0, air.conductivity,kSpace)
    kSpace = np.where(kSpace == 1, gas.conductivity,kSpace)
    kSpace = np.where(kSpace == 2, steel.conductivity,kSpace)
    kSpace = np.where(kSpace == 3, cordierite.conductivity,kSpace)

    # Assign specific heat capacities to each volume element's cordinate
    capacitySpace = np.where(capacitySpace == 0, air.capacity,capacitySpace)
    capacitySpace = np.where(capacitySpace == 1, gas.capacity,capacitySpace)
    i, j, k = np.where(capacitySpace == 2)
    for count in range(len(i)): # iterate through all steel volume elements
        iz = i[count]
        ix = j[count]
        iy = k[count]
        capacitySpace[iz, ix, iy] = steel.capacity(tempProfile[iz, ix, iy])
        capacitySpace = np.where(capacitySpace == 3, cordierite.capacity,capacitySpace)
    return kSpace, densitySpace, capacitySpace

def dTProfile(tempProfile):
    """
    DTPROFILE creates six 3D numpy arrays that record the temperature differenc
    e
    in the six relative directions for each volume element. The coordinates
    (position) of these values in these arrays then correspond to the position
    of the volume elements in MATERIALSPACE.
    """
    # Initialise the arrays

    dT1 = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # leftwards (inwards) (-y)
    dT2 = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # rightwards (outwards) (+y)
    dT3 = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # upwards (outwards) (-x)
    dT4 = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # downwards (inwards) (+x)
    dT5 = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # backwards (upstream) (-z)
    dT6 = np.zeros((zspace + 1, xspace + 2, yspace + 2)) # frontwards (downstream) (+z)
    Tn = tempProfile.copy() # retrieve a copy of temperature profile

    # Iterate over all volume elements
    for ix in range (xspace + 2):
        for iy in range(yspace + 2):
            for iz in range (zspace + 1):
                T = Tn[iz, ix, iy]

                # Calculate the relative temperature differences
                if iy > 0:
                    dT1[iz, ix, iy] = Tn[iz, ix, iy - 1] - T
                else:
                    dT1[iz, ix, iy] = Tn[iz, ix, iy + 2] - Tn[iz, ix, iy + 1]
                if iy < yspace + 1:
                    dT2[iz, ix, iy] = Tn[iz, ix, iy + 1] - T
                    #dT2[iz, ix, iy] = 0 #thermal insulation
                else:
                    dT2[iz, ix, iy] = 0
                if ix > 0:
                    dT3[iz, ix, iy] = Tn[iz, ix - 1, iy] - T
                    #dT3[iz, ix, iy] = 0 #thermal insulation
                else:
                    dT3[iz, ix, iy] = 0
                if ix < xspace + 1:
                    dT4[iz, ix, iy] = Tn[iz, ix + 1, iy] - T
                else:
                    dT4[iz, ix, iy] = Tn[iz, ix - 2, iy] - Tn[iz, ix - 1, iy]
                if iz > 0:
                    dT5[iz, ix, iy] = Tn[iz - 1, ix, iy] - T
                else:
                    dT5[iz, ix, iy] = 0
                if iz < zspace: # calculate normaly if inside the system
                    dT6[iz, ix, iy] = Tn[iz + 1, ix, iy] - T
                else: # there is no heat flow outside of the pipe end
                    dT6[iz, ix, iy] = 0
    return dT1, dT2, dT3, dT4, dT5, dT6

def dQProfile(tempProfile, materialSpace, kSpace, dT1, dT2, dT3, dT4, dT5, dT6):
    """
    DQPROFILE creates a 3D numpy array with the respective change in he
    at for each
    volume element based on the temperature diferences with neigbhouring elemen
    ts.
    The coordinates (position) of these values in these arrays then correspo
    nd to
    the position of the volume elements in MATERIALSPACE.

    """
    # Initiliase heat sink/source array
    q = np.empty((zspace + 1, xspace + 2, yspace + 2))

    # Iterate over all volume elements
    for ix in range(xspace+2):
        for iy in range(yspace+2):
            for iz in range(zspace+1):
                # Retrieve conductivites (neighbouring/central volume elements)
                k0 = kSpace[iz, ix, iy] #centre
                centerMaterial = materialSpace[iz,ix,iy]
                if iy > 0:
                    k1 = kSpace[iz, ix, iy - 1]
                    # leftwards (inwards) (-y)
                    if materialSpace[iz,ix,iy-1] != centerMaterial:
                        if materialSpace[iz,ix,iy-1] + centerMaterial in range(2,5):
                            C1 = solidGasBoundaryC
                        else:
                            C1 = cordieriteSteelBoundaryC
                    else:
                        C1 = cordieriteSteelBoundaryC
                else:
                    k1 = k0
                    C1 = np.infty
                if iy < yspace + 1:
                    k2 = kSpace[iz, ix, iy + 1]
                    # rightwards (outwards) (+y)
                    if materialSpace[iz, ix, iy+1] != centerMaterial:
                        if materialSpace[iz, ix, iy+1] + centerMaterial in range(2,5):
                            C2 = solidGasBoundaryC
                        else:
                            C2 = cordieriteSteelBoundaryC
                    else:
                        C2 = cordieriteSteelBoundaryC
                else:
                    k2 = k0
                    C2 = np.infty
                if ix > 0:
                    k3 = kSpace[iz, ix - 1, iy]
                    # upwards (outwards) (-x)
                    if materialSpace[iz, ix-1, iy] != centerMaterial:
                        if materialSpace[iz, ix-1, iy] + centerMaterial in range(2,5):
                            C3 = solidGasBoundaryC
                        else:
                            C3 = cordieriteSteelBoundaryC
                    else:
                        C3 = cordieriteSteelBoundaryC
                else:
                    k3 = k0
                    C3 = np.infty
                if ix < xspace + 1:
                    k4 = kSpace[iz, ix + 1, iy]
                    # downwards (inwards) (+x)
                    if materialSpace[iz, ix+1, iy] != centerMaterial:
                        if materialSpace[iz, ix+1, iy] + centerMaterial in range(2,5):
                            C4 = solidGasBoundaryC
                        else:
                            C4 = cordieriteSteelBoundaryC
                    else:
                        C4 = cordieriteSteelBoundaryC
                else:
                    k4 = k0
                    C4 = np.infty
                if iz > 0:
                    k5 = kSpace[iz - 1, ix, iy]
                    # backwards (upstream) (-z)
                    if materialSpace[iz-1, ix, iy] != centerMaterial:
                        if materialSpace[iz-1, ix, iy] + centerMaterial in range(2,5):
                            C5 = solidGasBoundaryC
                        else:
                            C5 = cordieriteSteelBoundaryC
                    else:
                        C5 = cordieriteSteelBoundaryC
                else:
                    k5 = k0
                    C5 = np.infty
                if iz < zspace: # calculate normaly if inside the system
                    k6 = kSpace[iz + 1, ix, iy]
                    if materialSpace[iz + 1, ix, iy] != centerMaterial:
                        # frontwards (downstream) (+z)
                        if materialSpace[iz + 1, ix, iy] + centerMaterial in range(2,5):
                            C6 = solidGasBoundaryC
                        else:
                            C6 = cordieriteSteelBoundaryC
                    else:
                        C6 = cordieriteSteelBoundaryC
                else: # there is nothing outside of the pipe end
                    k6 = 0
                    C6 = 0
                r2 = X[xspace + 1 - ix, iy, iz] ** 2 + Y[xspace + 1 - ix, iy, iz] ** 2
                if r2 <= 28**2:
                    constant = faceL*1e-3/(k0*faceArea)
                    if iz < zspace:
                        q[iz,ix,iy] = (dT1[iz,ix,iy]/(constant+faceL*1e-3/(k1*faceArea)+1/C1)
                    + dT2[iz,ix,iy]/(constant+faceL*1e-3/(k2*faceArea)+1/C2)
                    + dT3[iz,ix,iy]/(constant+faceL*1e-3/(k3*faceArea)+1/C3)
                    + dT4[iz,ix,iy]/(constant+faceL*1e-3/(k4*faceArea)+1/C4)
                    + dT5[iz,ix,iy]/(constant+faceL*1e-3/(k5*faceArea)+1/C5)
                    + dT6[iz,ix,iy]/(constant+faceL*1e-3/(k6*faceArea)+1/C6))
                    else: # as dT6 = 0
                        q[iz,ix,iy] = (dT1[iz,ix,iy]/(constant+faceL*1e-3/(k1*faceArea)+1/C1)
                    + dT2[iz,ix,iy]/(constant+faceL*1e-3/(k2*faceArea)+1/C2)
                    + dT3[iz,ix,iy]/(constant+faceL*1e-3/(k3*faceArea)+1/C3)
                    + dT4[iz,ix,iy]/(constant+faceL*1e-3/(k4*faceArea)+1/C4)
                    + dT5[iz,ix,iy]/(constant+faceL*1e-3/(k5*faceArea)+1/C5))
                else:
                    q[iz,ix,iy] = 0
                    q = np.nan_to_num(q)

    return q

def thermalEquilibrium(tempProfile, pre_tempProfile, deltaTeq):
    """
    THERMALEQUILIBRIUM decides whether thermal equilibrium has been achie
    ved
    or not based on a temperature difference threshold between individual
    time steps. Returns 'True' if thermal equilibrium has been achieved.
    """
    tempProfileChange = tempProfile - pre_tempProfile # find temperature change
    thermalEq = np.all(tempProfileChange < deltaTeq) # all deltaT below a threshold
    return thermalEq

def unlikeNeighbourSpacexyz(materialSpace):
    """
    UNLIKENEIGHBOURSPACEXYZ returns a 3D array made of the number of unl
    ike
    neighbours (based on material) for each volume element. The coordinates
    (position) of these values in these spaces then correspond to the
    position of the volume elements in MATERIALSPACE.
    """
    # Initialise the array with 0
    unlikeNeighbourSpace = np.zeros((zspace + 1, xspace + 2, yspace + 2))

    # Iterate over all volume elements
    for ix in range (xspace + 2):
        for iy in range(yspace + 2):
            for iz in range (zspace + 1):
                n = 0
                N0 = materialSpace[iz, ix, iy] # centre
                if iy > 0:
                    N1 = materialSpace[iz, ix, iy - 1] # leftwards (inwards) (-y)
                else:
                    N1 = N0
                if iy < yspace+1:
                    N2 = materialSpace[iz, ix, iy + 1] # rightwards (outwards) (+y)
                else:
                    N2 = 0 #air
                if ix > 0:
                    N3 = materialSpace[iz, ix - 1, iy] # upwards (outwards) (-x)
                else:
                    N3 = 0 #air
                if ix < xspace + 1:
                    N4 = materialSpace[iz, ix + 1, iy] # downwards (inwards) (+x)
                else:
                    N4 = N0
                if iz > 0:
                    N5 = materialSpace[iz-1, ix, iy] # backwards (upstream) (-z)
                else:
                    N5 = 0
                if iz < zspace - 1: # calculate normaly if inside the system
                    N6 = materialSpace[iz+1, ix, iy]
                else: # there is nothing outside of the pipe end
                    N6 = N0
                if N1 != N0:
                    n += 1
                if N2 != N0:
                    n += 1
                if N3 != N0:
                    n += 1
                if N4 != N0:
                    n += 1
                if N5 != N0:
                    n += 1
                if N6 != N0:
                    n += 1
                unlikeNeighbourSpace[iz,ix,iy] = n
    return unlikeNeighbourSpace

def externalPipeSurface(tempProfile, unlikeNeighbourSpace):
    """
    EXTERNALPIPESURFACE calculates the temperature profile of the external su
    rface
    of the steel pipe by averaging over the entire surface simulated. The profile
    is then added to tempProfilePipe, which records the profile with time.
    """
    # Initalise the arrays
    tempSum = np.zeros((zspace, 1))
    tempNo = np.zeros((zspace, 1))

    # Iterate over all volume elements
    for iz in range(1, zspace + 1):
        for ix in range(1, xspace + 1):
            for iy in range(1, yspace + 1):
                r2 = X[xspace + 1 - ix, iy, iz] ** 2 + Y[xspace + 1 - ix, iy, iz] ** 2
                if r2 > pipe.innerRadius ** 2 and r2 < pipe.outerRadius ** 2: # interior
                    if unlikeNeighbourSpace[iz, ix, iy] > 0: # exposed elements
                        tempSum[iz - 1] += tempProfile[iz, ix, iy]
                        tempNo[iz - 1] += 1
                    else:
                        continue
                else:
                    continue

    # Average out over all volume elements at a given z-coordinate
    tempAverage = tempSum / tempNo

    tempProfilePipe = tempAverage
    return tempProfilePipe

def exposedCatalyst(materialSpace, unlikeNeighbourSpace):
    """

    EXPOSEDCATALYST returns an array of coordinates of the volume elements of
    catalyst exposed to the exhaust gas.
    """
    exposedElements = [] # intialise the list

    # Iterate over all volume elements inside the cordierite matrix
    for iz in range(zspace - elementLengthCordierite + 1, zspace + 1):
        for ix in range(1, xspace + 1):
            for iy in range(1, yspace + 1):
                gasNeighbours = 0
                condition1 = unlikeNeighbourSpace[iz, ix, iy] > 0
                condition2 = materialSpace[iz, ix, iy] == 3
                if condition1 and condition2: # exposed cordierite
                    if materialSpace[iz, ix, iy - 1] == 1:
                        gasNeighbours += 1
                    if materialSpace[iz, ix, iy + 1] == 1:
                        gasNeighbours += 1
                    if materialSpace[iz, ix - 1, iy] == 1:
                        gasNeighbours += 1
                    if materialSpace[iz, ix + 1, iy] == 1:
                        gasNeighbours += 1
                    if gasNeighbours > 0:
                        # for every exposed face add one face (pair of coordinates)
                        for i in range(gasNeighbours):
                        # adds the coordinates of the exposed volume element
                            exposedElements.append([iz, ix, iy])
    return np.array(exposedElements)

def activatedCatalyst(tempProfile, exposedElements):
    """
    ACTIVATEDCATALYST returns an array of the temperatures of all the activated
    catalyst volume elements.
    """
    activatedElements = [] # initiliase return array
    for zxy in exposedElements:
        T = tempProfile[zxy[0], zxy[1], zxy[2]] # retrieve temperature
        if T >= Tcatalyst: # check for activation
            activatedElements.append(T)
        else:
            continue
    return np.array(activatedElements)

def COadsorption(tempProfile, activatedElements, nCO_fresh):
    """
    COADSORPTION calculates the amount of CO molecules produced
    and subtracts it from the amount of CO molecules present
    in the incoming exhaust gas.
    """
    dnCO = 0
    for T in activatedElements: # iterate over all active elements
        # Calculate the total amount of CO moles adsorbed in one dt
        dnCO = - Constant * np.exp(- activationEnergy / (R * T)) * dA * 1e-6
    nCO = nCO_fresh + dnCO
    return nCO


def COlimit(nCO, nCOtimeProfile, stepsTaken):
    """
    COLIMIT checks whether CO concentration has fallen below a defined
    threshold. It returns 'True' once the desired concentration limit
    is reached. Additionally, it records the number of CO molecules
    downstream and updates it into nCOtimeProfile.
    """
    nCOtimeProfile.append((stepsTaken * dt, nCO))
    if nCO <= nCO_end:
        return True, nCOtimeProfile
    else:
        return False, nCOtimeProfile

def plotDownstreamSurface(tempProfile, timeStep):
    """
    PLOTDOWNSTREAMSURFACE plots the temperature profile of the downstrea
    m surface
    every downstreamSurfaceInterval steps and saves a heat map as .png file.

    """
    if timeStep % downstreamSurfaceInterval != 0:
        return
    else:
        plt.figure(0)
        data = tempProfile[zspace - elementLengthCordierite + 5, 1:-1, 1:-1]
        data = data - 273 # convert from kelvin to celcius
        heat_map = sns.heatmap(data, cmap = 'plasma', vmin = 25, vmax = 300)
        plt.title("time: {0:.4f} s".format(round(timeStep * dt, 4)))
        plt.suptitle("Cordierite's surface temperature profile", fontweight = 'bold')
        plt.xlabel("Distance from the centre in x (mm)")
        xAxis = np.arange(0, xspace + 1, step = 4)
        plt.xticks(xAxis, xAxis, rotation = 0)
        yAxis = np.arange(0, xspace + 1, step = 4)
        plt.yticks(np.flip(yAxis), yAxis, rotation=0)
        plt.ylabel("Distance from the centre in y (mm)")
        plotIndex = timeStep / downstreamSurfaceInterval
        plt.savefig("{0}-cordieriteTempProfile.png".format(round(plotIndex, 0)), dpi=300)
        plt.close(0)
    return

def plotPipeSurface(tempProfilePipe, stepsTaken):
    """
    PLOTPIPESURFACE plots the temperature profile along the pipe's external surf
    ace
    over time as a heatmap.
    """
    heat_map = sns.heatmap(np.flipud(tempProfilePipe), cmap = 'plasma')
    plt.title("Pipe's external surface temperature profile")
    plt.xlabel("Time (s)")
    maxTime = stepsTaken * dt
    xticks = np.linspace(0, np.ma.size(tempProfilePipe, axis = 1)-2, num = 6)
    xlabels = np.linspace(0, maxTime, num=6, dtype=float)

    for i in range(len(xlabels)):
        xlabels[i] = round(xlabels[i], 3)
    plt.xticks(xticks, xlabels, rotation = 0)
    yticks = np.linspace(0, np.ma.size(tempProfilePipe, axis = 0), num = 11)
    ylabels = np.linspace(100, 0, num=11, dtype=int)
    plt.yticks(yticks, ylabels, rotation=0)
    plt.ylabel("Length along the pipe (mm)")
    plt.savefig("images2/pipeTempProfile.png", dpi=300)
    return

def nCOplot(nCOtimeProfile):
    """
    NCOPLOT plots the number of CO molecules downstream of the catalyst over ti
    me.
    """
    data = np.array(nCOtimeProfile)
    plt.plot(data[:, 0], data[:, 1], 'b-')
    plt.xlabel('time taken (s)')
    plt.ylabel('number of CO molecules')
    plt.savefig("nCOtimeProfile", dpi=300)
    return

def main(nt):
    """
    MAIN is the main function that assembles all of the previous functions and
    simulates for the amount of steps assigned.
    """
    # Initialise the arrays (spaces)
    tempProfile = np.full((zspace + 1, xspace + 2, yspace + 2), float(T0))
    materialSpace = materialSpacexyz() # materials
    tempProfilePipe = np.full(((zspace,1)), T0) # temp. profile along the pipe's surface
    nCOtimeProfile = [(0,nCO_fresh)] # CO concentration over time
    plotDownstreamSurface(tempProfile, 0) # initial plot of temperature profile
    unlikeNeighbourSpace1 = unlikeNeighbourSpacexyz(materialSpace)
    equilibriumReached = False
    if nt > 0:
        for n in range(nt):
            print('for time:', n * dt)
            # Initialise gas at the start of the pipe
            tempProfile[0] = Tgas
            materialSpace[0] = np.ones((yspace + 2, xspace + 2))

            # Move gas - MASS TRANSFER
            newTProfile, newMProfile = gasFlow(tempProfile, materialSpace)

            # Initialise physical quantities
            temporaryArray = propertySpace(newTProfile, newMProfile)
            kSpace, densitySpace, capacitySpace = temporaryArray

            # Find temperature differences
            dT1, dT2, dT3, dT4, dT5, dT6 = dTProfile(newTProfile)

            # Find change in heat

            q = dt*dQProfile(newTProfile, newMProfile, kSpace, dT1, dT2, dT3, dT4, dT5, dT6) # * dt as it is in WATTS SO WE NEED TO CHANGE IT INTO JOULES
            #q = dQProfile(newTProfile, newMProfile, kSpace, dT1, dT2, dT3, dT4, dT5, dT6)
            Tn = np.copy(newTProfile)
            # Update temperature profile
            tempProfile = Tn + (q / (densitySpace * volumeOfElement * capacitySpace))
            materialSpace = newMProfile
            #print('density, volmeoflement, cpacityspace, Tn')
            #print(densitySpace[95, 2, 3], volumeOfElement, capacitySpace[95, 2, 3], Tn[, 2, 3])
            #print('T, Q, dT1 (left), dT2 (right), dT3 (up), dT4 (down), dT5 (front), dT6 (back)')
            #print(tempProfile[95, 2, 3]) # q[95, 2, 3], dT1[95, 2, 3], dT2[95, 2, 3], dT3[9, 2, 3], dT4[95, 2, 3], dT5[95, 2, 3], dT6[95, 2, 3])
            #print('test', Tn[95, 2, 3] + (q[95, 2, 3] / (densitySpace[95, 2, 3] * volumeOfElement * capacitySpace[95, 2, 3], )))
            #print('upper layer')
            print(tempProfile[95, 2, :])
            plotDownstreamSurface(tempProfile, n + 1)

            #update tempProfilePipe
            tempProfilePipeNew = externalPipeSurface(tempProfile, unlikeNeighbourSpace1)
            tempProfilePipe = (np.append(tempProfilePipe, tempProfilePipeNew,axis=1))

            exposedElements = exposedCatalyst(materialSpace, unlikeNeighbourSpace1)
            activatedElements = activatedCatalyst(tempProfile, exposedElements)
            nCO = COadsorption(tempProfile, activatedElements, nCO_fresh)
            reduction, nCOtimeProfile = COlimit(nCO, nCOtimeProfile, n + 1)
            if activatedElements != []:
                if reduction == True:
                    print ('time taken for nCO to reduce to 0.01% is')
                    print (dt*n)
                    break
                else:
                    pass
            input1 = tempProfile[1:zspace+1,1:-1,1:-1]
            input2 = Tn[1:zspace+1,1:-1,1:-1]
            thermalEq = thermalEquilibrium(input1, input2, deltaTeq)
            if equilibriumReached == False:
                if thermalEq == True:
                    plotPipeSurface(tempProfilePipe, n + 1)
                    print ('is thermal equilibirum reached?')
                    print ('steps taken for equilibrium is')
                    print (n)
                    equilibriumReached = True
                else:
                    continue
            else:
                continue

    plt.figure(0)
    plt.contourf(tempProfile[1])
    plt.figure(1)
    plt.contourf(tempProfile[95])
    return tempProfile,materialSpace,tempProfilePipe,activatedElements,nCOtimeProfile

nt = 50000
tempProfile, materialSpace, tempProfilePipe, activatedElements, nCOtimeProfile0 = main(nt)
