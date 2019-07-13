# Import CUDA
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

# kernel
kernel_3x3 = SourceModule("""

__device__ unsigned getoff(unsigned &off){
  unsigned ret = off & 0x0F;
  off >>= 4;
  return ret;
}

// in-place is acceptable i.e. out == in)
// T = float or double only
const int block_size = 288;
typedef double T; // *** can set to float or double
__global__ void inv3x3(const T * __restrict__ in, T * __restrict__ out,
                       const size_t n, const unsigned * __restrict__ pat){

  __shared__ T si[block_size];
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  T det = 1;
  if (idx < n*9)
    det = in[idx];
  unsigned sibase = (threadIdx.x / 9)*9;
  unsigned lane = threadIdx.x - sibase; // cheaper modulo
  si[threadIdx.x] = det;
  __syncthreads();
  unsigned off = pat[lane];
  T a  = si[sibase + getoff(off)];
  a   *= si[sibase + getoff(off)];
  T b  = si[sibase + getoff(off)];
  b   *= si[sibase + getoff(off)];
  a -= b;
  __syncthreads();
  if (lane == 0) si[sibase+3] = a;
  if (lane == 3) si[sibase+4] = a;
  if (lane == 6) si[sibase+5] = a;
  __syncthreads();
  det =  si[sibase]*si[sibase+3]+si[sibase+1]*si[sibase+4]+si[sibase+2]*si[sibase+5];
  if (idx < n*9)
    out[idx] = a / det;
}   
""")

# host code
def gpuinv3x3(inp, n):
    # internal constants not to be modified
    hpat = (0x07584, 0x08172, 0x04251, 0x08365, 0x06280, 0x05032, 0x06473, 0x07061, 0x03140)
    # Convert parameters into numpy array
    # *** change next line between float32 and float64 to match float or double
    inpd = np.array(inp, dtype=np.float64)
    hpatd = np.array(hpat, dtype=np.uint32)
    # *** change next line between float32 and float64 to match float or double
    output = np.empty((n*9), dtype= np.float64)
    # Get kernel function
    matinv3x3 = kernel_3x3.get_function("inv3x3")
    # Define block, grid and compute
    blockDim = (288,1,1) # do not change
    gridDim = ((n/32)+1,1,1)
    # Kernel function
    matinv3x3 (
        cuda.In(inpd), cuda.Out(output), np.uint64(n), cuda.In(hpatd),
        block=blockDim, grid=gridDim)
    return output

# Covariance matrix 3x3
dimBlocks = 3

def buildObsCovarianceMatrix3_vec(k_d, mu_d, ir):

  # We fill the covariane matrix of observables 
  # as a function  of value of redshift in the 
  # interval 0.65 and 1.15 (see Table 2.4 page 22
  # on DESI paper.
  # 1) index t on rows and w on columns
  # 2) aa = bgs, bb = lrg, cc = elg, bc = lrg x elg

  # Array temp
  arrayCrossTemp = np.zeros((dimBlocks,dimBlocks,integ_prec,integ_prec))
  # rows indices
  z = zrange[ir]
  # right density
  global n_density
  n_density[ir,:] = dn3_array[ir,:]/dV_dz[ir]*Delta_z

  # Loop inside block
  for ub in range(dimBlocks):
    for vb in range(dimBlocks):

      # Diagonal terms
      if (ub == vb):

        # C_aaaa = 2 P_aa^2 N_a^2
	if (ub == 0):
          # N_a
	  N_a = (1+1./(n_density[ir,0]*P_obs_cross(k_d, mu_d, z, 
	        10**P_m(np.log10(k_d)), 10**P_m_NW(np.log10(k_d)), 
	        bias_array*sig_8_fid, growth_f[ir]*sig_8_fid, H_orig(z),
		H_orig(z), D_A_orig(z), D_A_orig(z),
		ir, 0, 0, None, None)))
          
	  arrayCrossTemp[ub][vb] = (2*P_obs_cross(k_d, mu_d, z, 
	                           10**P_m(np.log10(k_d)), 10**P_m_NW(np.log10(k_d)),
	                           bias_array*sig_8_fid, growth_f[ir]*sig_8_fid, H_orig(z),
				   H_orig(z), D_A_orig(z),
				   D_A_orig(z), ir, 0, 0, None, None)**2 * N_a**2)

        # C_bbbb = 2 P_bb^2 N_b^2
	elif (ub == 1):
          # N_b
	  N_b = (1+1./(n_density[ir,1]*P_obs_cross(k_d, mu_d, z, 
	        10**P_m(np.log10(k_d)), 10**P_m_NW(np.log10(k_d)), 
	        bias_array*sig_8_fid, growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z),
		D_A_orig(z), D_A_orig(z), ir, 1, 1, None, None)))

	  arrayCrossTemp[ub][vb] =  (2*P_obs_cross(k_d, mu_d, z, 10**P_m(np.log10(k_d)),
	                            10**P_m_NW(np.log10(k_d)), bias_array*sig_8_fid,
	                            growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z),
				    D_A_orig(z), ir, 1, 1, None, None)**2 * N_b**2)

        # C_abab = P_ab^2 + P_aa P_bb N_a N_b
	elif (ub == 2):
          # N_a
	  N_a = (1+1./(n_density[ir,0]*P_obs_cross(k_d, mu_d, z, 
	        10**P_m(np.log10(k_d)), 10**P_m_NW(np.log10(k_d)), bias_array*sig_8_fid,
	        growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z), D_A_orig(z), 
		ir, 0, 0, None, None)))

          # N_b
	  N_b = (1+1./(n_density[ir,1]*P_obs_cross(k_d, mu_d, z, 10**P_m(np.log10(k_d)), 
	        10**P_m_NW(np.log10(k_d)), bias_array*sig_8_fid,
	        growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z), D_A_orig(z),
		ir, 1, 1, None, None)))

	  arrayCrossTemp[ub][vb] = (P_obs_cross(k_d, mu_d, z, 10**P_m(np.log10(k_d)), 
	                           10**P_m_NW(np.log10(k_d)), bias_array*sig_8_fid,
	                           growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), 
				   D_A_orig(z), D_A_orig(z), ir, 0, 1, None, None)**2 + 
				   P_obs_cross(k_d, mu_d, z, 10**P_m(np.log10(k_d)), 
				   10**P_m_NW(np.log10(k_d)), bias_array*sig_8_fid, 
				   growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), 
				   D_A_orig(z), D_A_orig(z), ir, 0, 0, None, None) * 
				   P_obs_cross(k_d, mu_d, z, 10**P_m(np.log10(k_d)), 
				   10**P_m_NW(np.log10(k_d)), bias_array*sig_8_fid,
				   growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), 
				   D_A_orig(z), D_A_orig(z), ir, 1, 1, None, None) 
				   *  N_a * N_b)
      
      # Off-diagonal terms
      # C_aabb = 2 P_ab^2 (eq.21 White's)
      elif (ub == 0 and vb == 1):

	arrayCrossTemp[ub][vb] = (2 * P_obs_cross(k_ref, mu_ref, z, 10**P_m(np.log10(k_ref)),
	                         10**P_m_NW(np.log10(k_ref)), bias_array*sig_8_fid,
	                         growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), 
				 D_A_orig(z), D_A_orig(z), ir, 0, 1, None, None)**2)

	arrayCrossTemp[vb][ub] = arrayCrossTemp[ub][vb]
     
      # C_aaab = 2 P_ab P_aa N_a (eq.23 White's)
      elif (ub == 0 and vb == 2):

	# N_a
	N_a = (1+1./(n_density[ir,0]*P_obs_cross(k_ref, mu_ref, z, 10**P_m(np.log10(k_ref)), 
	      10**P_m_NW(np.log10(k_ref)), bias_array*sig_8_fid,
	      growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z),
	      D_A_orig(z), ir, 0, 0, None, None)))

	arrayCrossTemp[ub][vb] = (2 * P_obs_cross(k_ref, mu_ref, z, 10**P_m(np.log10(k_ref)), 
	                         10**P_m_NW(np.log10(k_ref)), bias_array*sig_8_fid,
	                         growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z),
				 D_A_orig(z), ir, 0, 1, None, None) * 
				 P_obs_cross(k_ref, mu_ref, z, 10**P_m(np.log10(k_ref)), 
				 10**P_m_NW(np.log10(k_ref)), bias_array*sig_8_fid,
				 growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z),
				 D_A_orig(z), ir, 0, 0, None, None) * N_a)

	arrayCrossTemp[vb][ub] = arrayCrossTemp[ub][vb]
     
      # C_bbab = C_abbb = 2 P_ab P_bb (eq.20 White's)
      elif (ub == 1 and vb == 2):

	arrayCrossTemp[ub][vb] = (2 * P_obs_cross(k_ref, mu_ref, z, 10**P_m(np.log10(k_ref)),
	                         10**P_m_NW(np.log10(k_ref)), bias_array*sig_8_fid,
	                         growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z),
				 D_A_orig(z), ir, 0, 1, None, None) * 
				 P_obs_cross(k_ref, mu_ref, z, 10**P_m(np.log10(k_ref)),
				 10**P_m_NW(np.log10(k_ref)), bias_array*sig_8_fid,
				 growth_f[ir]*sig_8_fid, H_orig(z), H_orig(z), D_A_orig(z),
				 D_A_orig(z), ir, 1, 1, None, None))

	arrayCrossTemp[vb][ub] = arrayCrossTemp[ub][vb]

  return arrayCrossTemp

# Cross section : invert integ_prec*integ_prec 3x3 matrices
switch = 'cross'

# Declaration of inverse cross matrix
invCrossMatrix = np.zeros((dimBlocks,dimBlocks,integ_prec,integ_prec))

# Declaration of inverse cross matrix
invCrossMatrix_temp = np.zeros((integ_prec**2,dimBlocks,dimBlocks))

# Create arrayFullCross_vec array
arrayFullCross_vec = np.zeros((dimBlocks,dimBlocks,integ_prec,integ_prec))

# Create arrayFullCross_vec array
invCrossMatrix_gpu = np.zeros((dimBlocks*dimBlocks*(integ_prec**2)))

# Build observables covariance matrix
arrayFullCross_vec = buildObsCovarianceMatrix3_vec(k_ref, mu_ref, ir)

# Performing batch inversion 3x3
invCrossMatrix_gpu = gpuinv3x3(arrayFullCross_vec.flatten('F'),integ_prec**2)

# New reshape
invCrossMatrix_temp = invCrossMatrix_gpu.reshape(integ_prec**2,dimBlocks,dimBlocks)
# Final reshape : don't forget ".T" transpose operator
invCrossMatrix = (invCrossMatrix_temp.reshape(integ_prec,integ_prec,dimBlocks,dimBlocks)).T

