import numpy as np

def getSimulationParams1P1D(Lx, Ly, Lz):
  xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  if Lx < 256:
    blockX = Lx
  else: 
    blockX = 256
    gridX = np.floor(Lx/256)
  xSize, ySize, zSize = blockX*gridX, blockY*gridY, blockZ*gridZ
  return xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz

def getSimulationParams1P2D(Lx, Ly, Lz):
  xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  if Lx < 16:
    blockX = Lx
  else: 
    blockX = 16
    gridX = np.floor(Lx/16)
  if Ly < 16:
    blockY = Ly
  else: 
    blockY = 16
    gridY = np.floor(Ly/16)
  xSize, ySize, zSize = blockX*gridX, blockY*gridY, blockZ*gridZ
  return xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz

def getSimulationParams1P3D(Lx, Ly, Lz):
  xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  if Lx < 8:
    blockX = Lx
  else: 
    blockX = 8
    gridX = np.floor(Lx/8)
  if Ly < 8:
    blockY = Ly
  else: 
    blockY = 8
    gridY = np.floor(Ly/8)
  if Lz < 4:
    blockZ = Lz
  else: 
    blockZ = 4
    gridZ = np.floor(Lz/4)
  xSize, ySize, zSize = blockX*gridX, blockY*gridY, blockZ*gridZ
  return xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz


def getSimulationParams2P1D(Lx, Ly, Lz, num_GPUs):
  xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  Nx = 1
  blockX = 64
  Qx = 2*blockX*num_GPUs*Nx 
  while Qx<2*Lx:
    Nx += 1
    Qx = 2*blockX*num_GPUs*Nx
  gridX = Nx*(Qx-1)
  xSize, ySize, zSize = blockX*gridX, blockY*gridY, blockZ*gridZ
  return xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz

def getSimulationParams2P2D(Lx, Ly, Lz, num_GPUs):
  xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  Nx, Ny = 1, 1
  blockX = 8
  blockY = 8
  Qx = 2*blockX*num_GPUs*Nx
  Qy = 2*blockY*Ny
  while Qx<2*Lx:
    Nx += 1
    Qx = 2*blockX*num_GPUs*Nx
  while Qy<2*Ly:
    Ny += 1
    Qy = 2*blockY*Ny
  gridX = Nx*(Qx-1)
  gridY = Ny*(Qy-1)
  xSize, ySize, zSize = blockX*gridX, blockY*gridY, blockZ*gridZ
  return xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz

def getSimulationParams2P3D(Lx, Ly, Lz, num_GPUs):
  xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  Nx, Ny, Nz = 1, 1, 1
  blockX = 8
  blockY = 8
  blockZ = 4
  Qx = 2*blockX*num_GPUs*Nx
  Qy = 2*blockY*Ny
  Qz = 2*blockY*Nz
  while Qx<2*Lx:
    Nx += 1
    Qx = 2*blockX*num_GPUs*Nx
  while Qy<2*Ly:
    Ny += 1
    Qy = 2*blockY*Ny
  while Qz<2*Lz:
    Nz += 1
    Qz = 2*blockY*Nz
  gridX = Nx*(Qx-1)
  gridY = Ny*(Qy-1)
  gridZ = Nz*(Qz-1)
  xSize, ySize, zSize = blockX*gridX, blockY*gridY, blockZ*gridZ
  return xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz
