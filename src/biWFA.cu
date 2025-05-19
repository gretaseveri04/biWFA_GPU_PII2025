extern "C" {
	#include "wavefront/wavefront_align.h"
}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <fstream>
#include <vector>
#include "headers/commons.h"
#include "headers/biWFA.h"
#include <chrono>

#define CHECK(call)                                                                     \
{                                                                                     \
	const cudaError_t err = call;                                                     \
	if (err != cudaSuccess)                                                           \
	{                                                                                 \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(EXIT_FAILURE);                                                           \
	}                                                                                 \
}

#define CHECK_KERNELCALL()                                                            \
{                                                                                     \
	const cudaError_t err = cudaGetLastError();                                       \
	if (err != cudaSuccess)                                                           \
	{                                                                                 \
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(EXIT_FAILURE);                                                           \
	}                                                                                 \
}

__device__ void extend_max(bool *finish, const int score, int32_t *max_ak, wf_components_t *wf, const int max_score_scope, const int alignment_k, const int32_t alignment_offset, const int pattern_len) {
    if (wf->mwavefronts[score%num_wavefronts].offsets == NULL) {
        if (wf->alignment.num_null_steps > max_score_scope) {
            *finish = true;
        } else {
            *finish = false;
        }
    } else {
        // wavefront_extend_matches_packed_end2end_max()
        bool end_reached = false;
        int32_t max_antidiag_loc = 0;
        
        // Iterate over all wavefront offsets
        int k_start = wf->mwavefronts[score%num_wavefronts].lo;
        int k_end = wf->mwavefronts[score%num_wavefronts].hi;
        
        for (int k = k_start; k <= k_end; ++k) {
            int32_t offset = wf->mwavefronts[score%num_wavefronts].offsets[k];
            if (offset == OFFSET_NULL) continue;
            
            // wavefront_extend_matches_kernel_blockwise() or wavefront_extend_matches_kernel()
            int equal_chars = 0;
            for (int i = offset; i < pattern_len; i++) {
                if((i - k) >= 0 && (i - k) < pattern_len) {
                    if (wf->alignment.pattern[i - k] == wf->alignment.text[i]) {
                        equal_chars++;
                    } else break;
                }
            }
            offset += equal_chars;
            
            // Return extended offset
            wf->mwavefronts[score%num_wavefronts].offsets[k] = offset;
            
            // Calculate antidiagonal and update max if needed
            int32_t antidiag = (2 * offset) - k;
            if (max_antidiag_loc < antidiag) {
                max_antidiag_loc = antidiag;
            }
        }
        
        // Update the max antidiagonal location
        *max_ak = max_antidiag_loc;
        
        // wavefront_termination_end2end()
        if (wf->mwavefronts[score%num_wavefronts].lo > alignment_k || alignment_k > wf->mwavefronts[score%num_wavefronts].hi) {
            end_reached = false;
        } else {
            int32_t moffset = wf->mwavefronts[score%num_wavefronts].offsets[alignment_k];
            if (moffset < alignment_offset) {
                end_reached = false;
            } else {
                end_reached = true;
            }
        }
        
        *finish = end_reached;
    }
}

__device__ void extend(bool *finish, const int score, const wf_components_t *wf, const int max_score_scope, const int alignment_k, const int32_t alignment_offset, const int pattern_len) {
    wf_t *mwf = &wf->mwavefronts[score % num_wavefronts];
    
    if (mwf->offsets == NULL) {
        *finish = (wf->alignment.num_null_steps > max_score_scope);
        return;
    }

    int lo = mwf->lo;
    int hi = mwf->hi;
    int k = lo + threadIdx.x;

    int32_t offset = 0;
    if (k <= hi) {
        offset = mwf->offsets[k];

        for (int i = offset; i < pattern_len; ++i) {
            int pattern_pos = i - k;
            int text_pos = i;

            if (pattern_pos < 0 || pattern_pos >= pattern_len) break;
            if (wf->alignment.pattern[pattern_pos] != wf->alignment.text[text_pos]) break;

            ++offset;
        }

        mwf->offsets[k] = offset;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        bool end_reached = false;
        if (alignment_k >= lo && alignment_k <= hi) {
            int32_t moffset = mwf->offsets[alignment_k];
            end_reached = (moffset >= alignment_offset);
        }
        *finish = end_reached;
    }
}

__device__ void nextWF(int *score, wf_components_t *wf, const bool forward, const int max_score_scope, const int text_len, const int pattern_len, int32_t *matrix_wf_m_g, int32_t *matrix_wf_i_g, int32_t *matrix_wf_d_g) {
    // Compute next (s+1) wavefront
    ++(*score);

    int score_mod = *score%num_wavefronts;

    // wavefront_compute_affine()
    int mismatch = *score - penalty_mismatch;
    int gap_open = *score - penalty_gap_open - penalty_gap_ext;
    int gap_extend = *score - penalty_gap_ext;

    // wavefront_compute_get_mwavefront()
    if((*score / num_wavefronts) > 0) {
        // Resetting old wavefronts' values
        wf->mwavefronts[score_mod].lo = -1;
        wf->mwavefronts[score_mod].hi = 1;
        wf->iwavefronts[score_mod].lo = -1;
        wf->iwavefronts[score_mod].hi = 1;
        wf->dwavefronts[score_mod].lo = -1;
        wf->dwavefronts[score_mod].hi = 1;
    }
    wf->mwavefronts[score_mod].offsets = matrix_wf_m_g + (num_wavefronts * wf_length * blockIdx.x) + (score_mod * wf_length) + wf_length/2;
    wf->mwavefronts[score_mod].null = false;
    wf->iwavefronts[score_mod].offsets = matrix_wf_i_g + (num_wavefronts * wf_length * blockIdx.x) + (score_mod * wf_length) + wf_length/2;
    wf->iwavefronts[score_mod].null = false;
    wf->dwavefronts[score_mod].offsets = matrix_wf_d_g + (num_wavefronts * wf_length * blockIdx.x) + (score_mod * wf_length) + wf_length/2;
    wf->dwavefronts[score_mod].null = false;

    wf_t in_mwavefront_misms = (mismatch < 0 || wf->mwavefronts[mismatch%num_wavefronts].offsets == NULL || wf->mwavefronts[mismatch%num_wavefronts].null) ? wf->wavefront_null : wf->mwavefronts[mismatch%num_wavefronts];
    wf_t in_mwavefront_open = (gap_open < 0 || wf->mwavefronts[gap_open%num_wavefronts].offsets == NULL || wf->mwavefronts[gap_open%num_wavefronts].null) ? wf->wavefront_null : wf->mwavefronts[gap_open%num_wavefronts];
    wf_t in_iwavefront_ext = (gap_extend < 0 || wf->iwavefronts[gap_extend%num_wavefronts].offsets == NULL || wf->iwavefronts[gap_extend%num_wavefronts].null) ? wf->wavefront_null : wf->iwavefronts[gap_extend%num_wavefronts];
    wf_t in_dwavefront_ext = (gap_extend < 0 || wf->dwavefronts[gap_extend%num_wavefronts].offsets == NULL || wf->dwavefronts[gap_extend%num_wavefronts].null) ? wf->wavefront_null : wf->dwavefronts[gap_extend%num_wavefronts];

    if (in_mwavefront_misms.null && in_mwavefront_open.null && in_iwavefront_ext.null && in_dwavefront_ext.null) {
        // wavefront_compute_allocate_output_null()
        wf->alignment.num_null_steps++; // Increment null-steps
        // Nullify Wavefronts
        wf->mwavefronts[score_mod].null = true;
        wf->iwavefronts[score_mod].null = true;
        wf->dwavefronts[score_mod].null = true;
    } else {
        wf->alignment.num_null_steps = 0;
        int hi, lo;

        // wavefront_compute_limits_input()
        int min_lo = in_mwavefront_misms.lo;
        int max_hi = in_mwavefront_misms.hi;

        if (!in_mwavefront_open.null && min_lo > (in_mwavefront_open.lo - 1)) min_lo = in_mwavefront_open.lo - 1;
        if (!in_mwavefront_open.null && max_hi < (in_mwavefront_open.hi + 1)) max_hi = in_mwavefront_open.hi + 1;
        if (!in_iwavefront_ext.null && min_lo > (in_iwavefront_ext.lo + 1)) min_lo = in_iwavefront_ext.lo + 1;
        if (!in_iwavefront_ext.null && max_hi < (in_iwavefront_ext.hi + 1)) max_hi = in_iwavefront_ext.hi + 1;
        if (!in_dwavefront_ext.null && min_lo > (in_dwavefront_ext.lo - 1)) min_lo = in_dwavefront_ext.lo - 1;
        if (!in_dwavefront_ext.null && max_hi < (in_dwavefront_ext.hi - 1)) max_hi = in_dwavefront_ext.hi - 1;
        lo = min_lo;
        hi = max_hi;

        // wavefront_compute_allocate_output()
        int effective_lo = lo;
        int effective_hi = hi;

        // wavefront_compute_limits_output()
        int eff_lo = effective_lo - (max_score_scope + 1);
        int eff_hi = effective_hi + (max_score_scope + 1);
        effective_lo = MIN(eff_lo, wf->alignment.historic_min_lo);
        effective_hi = MAX(eff_hi, wf->alignment.historic_max_hi);
        wf->alignment.historic_min_lo = effective_lo;
        wf->alignment.historic_max_hi = effective_hi;

        // Allocate M-Wavefront
        wf->mwavefronts[score_mod].lo = lo;
        wf->mwavefronts[score_mod].hi = hi;
        // Allocate I1-Wavefront
        if (!in_mwavefront_open.null || !in_iwavefront_ext.null) {
            wf->iwavefronts[score_mod].lo = lo;
            wf->iwavefronts[score_mod].hi = hi;
        } else {
            wf->iwavefronts[score_mod].null = true;
        }
        // Allocate D1-Wavefront
        if (!in_mwavefront_open.null || !in_dwavefront_ext.null) {
            wf->dwavefronts[score_mod].lo = lo;
            wf->dwavefronts[score_mod].hi = hi;
        } else {
            wf->dwavefronts[score_mod].null = true;
        }

        // wavefront_compute_init_ends()
        // Init wavefront ends
        bool m_misms_null = in_mwavefront_misms.null;
        bool m_gap_null = in_mwavefront_open.null;
        bool i_ext_null = in_iwavefront_ext.null;
        bool d_ext_null = in_dwavefront_ext.null;

        if (!m_misms_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_mwavefront_misms.wf_elements_init_max >= hi) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_mwavefront_misms.wf_elements_init_max, in_mwavefront_misms.hi);
                int k;
                int tidx = threadIdx.x; 
                int num_threads = blockDim.x; 

                for (int k = max_init + 1 + tidx; k <= hi; k += num_threads) {
                    in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                }

                if (tidx == 0) {
                    in_mwavefront_misms.wf_elements_init_max = hi;
                }
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_mwavefront_misms.wf_elements_init_min <= lo) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_mwavefront_misms.wf_elements_init_min, in_mwavefront_misms.lo);
                int k;
                int tidx = threadIdx.x; 
                int num_threads = blockDim.x; 

                for (int k = lo + tidx; k < min_init; k += num_threads) {
                    in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                }
                if (tidx == 0) {
                    in_mwavefront_misms.wf_elements_init_min = lo;
                }

            }
        }
        if (!m_gap_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_mwavefront_open.wf_elements_init_max >= hi + 1) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_mwavefront_open.wf_elements_init_max, in_mwavefront_open.hi);
                int k;
                int tidx = threadIdx.x; 
                int num_threads = blockDim.x; 

                for (int k = max_init + 1 + tidx; k <= hi + 1; k += num_threads) {
                    in_mwavefront_open.offsets[k] = OFFSET_NULL;
                }
                if (tidx == 0) {
                    in_mwavefront_open.wf_elements_init_max = hi + 1;
                }

            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_mwavefront_open.wf_elements_init_min <= lo - 1) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_mwavefront_open.wf_elements_init_min, in_mwavefront_open.lo);
                int k;
                int tidx = threadIdx.x;
                int num_threads = blockDim.x;

                for (int k = lo - 1 + tidx; k < min_init; k += num_threads) {
                    in_mwavefront_open.offsets[k] = OFFSET_NULL;
                }

                if (tidx == 0) {
                    in_mwavefront_open.wf_elements_init_min = lo - 1;
                }

            }
        }
        if (!i_ext_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_iwavefront_ext.wf_elements_init_max >= hi) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_iwavefront_ext.wf_elements_init_max, in_iwavefront_ext.hi);
                int k;
                int tidx = threadIdx.x;
                int num_threads = blockDim.x;

                for (int k = max_init + 1 + tidx; k <= hi; k += num_threads) {
                    in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                }

                if (tidx == 0) {
                    in_iwavefront_ext.wf_elements_init_max = hi;
                }

            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_iwavefront_ext.wf_elements_init_min <= lo - 1) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_iwavefront_ext.wf_elements_init_min, in_iwavefront_ext.lo);
                int k;
                int tidx = threadIdx.x;
                int num_threads = blockDim.x;

                for (int k = lo - 1 + tidx; k < min_init; k += num_threads) {
                    in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                }

                if (tidx == 0) {
                    in_iwavefront_ext.wf_elements_init_min = lo - 1;
                }

            }
        }
        if (!d_ext_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_dwavefront_ext.wf_elements_init_max >= hi + 1) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_dwavefront_ext.wf_elements_init_max, in_dwavefront_ext.hi);
                int k;
                int tidx = threadIdx.x;
                int num_threads = blockDim.x;

                for (int k = max_init + 1 + tidx; k <= hi + 1; k += num_threads) {
                    in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                }

                if (tidx == 0) {
                    in_dwavefront_ext.wf_elements_init_max = hi + 1;
                }

            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_dwavefront_ext.wf_elements_init_min <= lo) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_dwavefront_ext.wf_elements_init_min, in_dwavefront_ext.lo);
                int k;
                int tidx = threadIdx.x;
                int num_threads = blockDim.x;
                
                for (int k = lo + tidx; k < min_init; k += num_threads) {
                    in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                
                if (tidx == 0) {
                    in_dwavefront_ext.wf_elements_init_min = lo;
                }

            }
        }

        //wavefront_compute_affine_idm()
        // Compute-Next kernel loop
        int tidx = threadIdx.x;
        for (int i = lo; i <= hi; i += blockDim.x) {
            int idx = tidx + i;
            if (idx <= hi) {
                // Update I1
                int32_t ins_o = in_mwavefront_open.offsets[idx - 1];
                int32_t ins_e = in_iwavefront_ext.offsets[idx - 1];
                int32_t ins = MAX(ins_o, ins_e) + 1;
                wf->iwavefronts[score_mod].offsets[idx] = ins;

                // Update D1
                int32_t del_o = in_mwavefront_open.offsets[idx + 1];
                int32_t del_e = in_dwavefront_ext.offsets[idx + 1];
                int32_t del = MAX(del_o, del_e);
                wf->dwavefronts[score_mod].offsets[idx] = del;

                // Update M
                int32_t misms = in_mwavefront_misms.offsets[idx] + 1;
                int32_t max = MAX(del, MAX(misms, ins));

                // Adjust offset out of boundaries
                uint32_t h = max;
                uint32_t v = max - idx;
                if (h > text_len) max = OFFSET_NULL;
                if (v > pattern_len) max = OFFSET_NULL;
                wf->mwavefronts[score_mod].offsets[idx] = max;
            }
        }

        // wavefront_compute_process_ends()
        if (wf->mwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->mwavefronts[score_mod].lo;
            for (k = wf->mwavefronts[score_mod].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[score_mod].hi = k; // Set new hi
            wf->mwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->mwavefronts[score_mod].hi;
            for (k = wf->mwavefronts[score_mod].lo ; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[score_mod].lo = k; // Set new lo
            wf->mwavefronts[score_mod].wf_elements_init_min = k;
            wf->mwavefronts[score_mod].null = (wf->mwavefronts[score_mod].lo > wf->mwavefronts[score_mod].hi);
        }
        if (wf->iwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->iwavefronts[score_mod].lo;
            for (k = wf->iwavefronts[score_mod].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[score_mod].hi = k; // Set new hi
            wf->iwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->iwavefronts[score_mod].hi;
            for (k = wf->iwavefronts[score_mod].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[score_mod].lo = k; // Set new lo
            wf->iwavefronts[score_mod].wf_elements_init_min = k;
            wf->iwavefronts[score_mod].null = (wf->iwavefronts[score_mod].lo > wf->iwavefronts[score_mod].hi);
        }
        if (wf->dwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->dwavefronts[score_mod].lo;
            for (k = wf->dwavefronts[score_mod].hi; k >= lo ; --k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[score_mod].hi = k; // Set new hi
            wf->dwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->dwavefronts[score_mod].hi;
            for (k = wf->dwavefronts[score_mod].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[score_mod].lo = k; // Set new lo
            wf->dwavefronts[score_mod].wf_elements_init_min = k;
            wf->dwavefronts[score_mod].null = (wf->dwavefronts[score_mod].lo > wf->dwavefronts[score_mod].hi);
        }
    }
}

__device__ void breakpoint_indel2indel(const int score_0, const int score_1, const wf_t *dwf_0, const wf_t *dwf_1, int *breakpoint_score, const int text_len, const int pattern_len) {
    int lo_0 = dwf_0->lo;
    int hi_0 = dwf_0->hi;
    int lo_1 = text_len - pattern_len - dwf_1->hi;
    int hi_1 = text_len - pattern_len - dwf_1->lo;

    if (hi_1 < lo_0 || hi_0 < lo_1) return;

    int min_hi = min(hi_0, hi_1);
    int max_lo = max(lo_0, lo_1);

    __shared__ int local_min[NUM_THREADS];
    int tid = threadIdx.x;
    local_min[tid] = INT_MAX;

    for (int k_0 = max_lo + tid; k_0 <= min_hi; k_0 += NUM_THREADS) {
        int k_1 = text_len - pattern_len - k_0;
        int dh_0 = dwf_0->offsets[k_0];
        int dh_1 = dwf_1->offsets[k_1];

        if ((dh_0 + dh_1) >= text_len) {
            int candidate = score_0 + score_1 - penalty_gap_open;
            if (candidate < local_min[tid]) {
                local_min[tid] = candidate;
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        int min_val = INT_MAX;
        for (int i = 0; i < NUM_THREADS; i++) {
            if (local_min[i] < min_val) {
                min_val = local_min[i];
            }
        }

        if (min_val < *breakpoint_score) {
            *breakpoint_score = min_val;
        }
    }
}

__device__ void breakpoint_m2m(const int score_0, const int score_1, const wf_t *mwf_0, const wf_t *mwf_1, int *breakpoint_score, const int text_len, const int pattern_len) {
    // Check wavefronts overlapping
    int lo_0 = mwf_0->lo;
    int hi_0 = mwf_0->hi;
    int lo_1 = text_len - pattern_len - mwf_1->hi;
    int hi_1 = text_len - pattern_len - mwf_1->lo;

    if (hi_1 < lo_0 || hi_0 < lo_1) return;
    
    // Compute overlapping interval
    int min_hi = MIN(hi_0, hi_1);
    int max_lo = MAX(lo_0, lo_1);
    int k_0;
    for (k_0 = max_lo; k_0 <= min_hi; k_0++) {
        const int k_1 = text_len - pattern_len - k_0;
        // Fetch offsets
        const int mh_0 = mwf_0->offsets[k_0];
        const int mh_1 = mwf_1->offsets[k_1];
        // Check breakpoint m2m
        if (mh_0 + mh_1 >= text_len && score_0 + score_1 < *breakpoint_score) {
            *breakpoint_score = score_0 + score_1; 
            return;
        }
    }
}

__device__ void overlap(const int score_0, const wf_components_t *wf_0, const int score_1, const wf_components_t *wf_1, const int max_score_scope, int *breakpoint_score, const int text_len, const int pattern_len) {
    int score_mod_0 = score_0 % num_wavefronts;
    wf_t *mwf_0 = &wf_0->mwavefronts[score_mod_0];

    if (mwf_0 == NULL) return;
    wf_t *d1wf_0 = &wf_0->dwavefronts[score_mod_0];
    wf_t *i1wf_0 = &wf_0->iwavefronts[score_mod_0];

    int i;
    for (i = 0; i < max_score_scope; ++i) {
        const int score_i = score_1 - i;
        if (score_i < 0) break;
        int score_mod_i = score_i % num_wavefronts;

        if (score_0 + score_i - penalty_gap_open >= *breakpoint_score) continue;

        wf_t *d1wf_1 = &wf_1->dwavefronts[score_mod_i];
        if (d1wf_0 != NULL && d1wf_1 != NULL) {
            breakpoint_indel2indel(score_0, score_i, d1wf_0, d1wf_1, breakpoint_score, text_len, pattern_len);
        }

        wf_t *i1wf_1 = &wf_1->iwavefronts[score_mod_i];
        if (i1wf_0 != NULL && i1wf_1 != NULL) {
            breakpoint_indel2indel(score_0, score_i, i1wf_0, i1wf_1, breakpoint_score, text_len, pattern_len);
        }

        if (score_0 + score_i >= *breakpoint_score) continue;
        wf_t *mwf_1 = &wf_1->mwavefronts[score_mod_i];

        if (mwf_1 != NULL && mwf_0->offsets != NULL && mwf_1->offsets != NULL) {
            if (mwf_0->lo <= mwf_0->hi && mwf_1->lo <= mwf_1->hi) {
                breakpoint_m2m(score_0, score_i, mwf_0, mwf_1, breakpoint_score, text_len, pattern_len);
            }
        }
    }
}


__global__ void biWFA_kernel(char *pattern_concat_g, char *text_concat_g, char *pattern_r_concat_g, char *text_r_concat_g, int *pattern_lengths_g, int *text_lengths_g,
    int *pattern_offsets_g, int *text_offsets_g, int *breakpoint_score_g, wf_t *mwavefronts_f, wf_t *iwavefronts_f, wf_t *dwavefronts_f, wf_t *mwavefronts_r, wf_t *iwavefronts_r,
    wf_t *dwavefronts_r, const int lo_g, const int hi_g, int32_t *offsets_g, const int max_score_scope, int32_t *matrix_wf_m_f, int32_t *matrix_wf_i_f, int32_t *matrix_wf_d_f,
    int32_t *matrix_wf_m_r, int32_t *matrix_wf_i_r, int32_t *matrix_wf_d_r) 
{
    int alignment_id = blockIdx.x;
    int lo = lo_g;
    int hi = hi_g;

    int pattern_offset = pattern_offsets_g[alignment_id];
    int text_offset = text_offsets_g[alignment_id];
    int pattern_len = pattern_lengths_g[alignment_id];
    int text_len = text_lengths_g[alignment_id];

    char *pattern_f = pattern_concat_g + pattern_offset;
    char *text_f = text_concat_g + text_offset;
    char *pattern_r = pattern_r_concat_g + pattern_offset;
    char *text_r = text_r_concat_g + text_offset;

    int total_offsets_size = num_wavefronts * wf_length;
    for (int i = threadIdx.x; i < total_offsets_size; i += blockDim.x) {
        int base_idx = alignment_id * total_offsets_size + i;
        matrix_wf_m_f[base_idx] = OFFSET_NULL;
        matrix_wf_i_f[base_idx] = OFFSET_NULL;
        matrix_wf_d_f[base_idx] = OFFSET_NULL;
        matrix_wf_m_r[base_idx] = OFFSET_NULL;
        matrix_wf_i_r[base_idx] = OFFSET_NULL;
        matrix_wf_d_r[base_idx] = OFFSET_NULL;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_wavefronts; i += blockDim.x) {
        int base_idx = alignment_id * num_wavefronts + i;
        wf_t *wfs[] = { &mwavefronts_f[base_idx], &iwavefronts_f[base_idx], &dwavefronts_f[base_idx],
                        &mwavefronts_r[base_idx], &iwavefronts_r[base_idx], &dwavefronts_r[base_idx] };
        for (int w = 0; w < 6; ++w) {
            wfs[w]->null = true;
            wfs[w]->lo = 0;
            wfs[w]->hi = 0;
            wfs[w]->offsets = NULL;
            wfs[w]->wf_elements_init_min = 0;
            wfs[w]->wf_elements_init_max = 0;
        }
    }
    __syncthreads();

    wf_components_t wf_f, wf_r;
    wf_alignment_t alignment_f, alignment_r;

    alignment_f.pattern = pattern_f;
    alignment_f.text = text_f;
    alignment_f.historic_max_hi = hi;
    alignment_f.historic_min_lo = lo;
    alignment_f.num_null_steps = 0;
    wf_f.alignment = alignment_f;

    alignment_r.pattern = pattern_r;
    alignment_r.text = text_r;
    alignment_r.historic_max_hi = hi;
    alignment_r.historic_min_lo = lo;
    alignment_r.num_null_steps = 0;
    wf_r.alignment = alignment_r;

    int wf_base_idx = num_wavefronts * alignment_id;
    wf_f.mwavefronts = mwavefronts_f + wf_base_idx;
    wf_f.iwavefronts = iwavefronts_f + wf_base_idx;
    wf_f.dwavefronts = dwavefronts_f + wf_base_idx;
    wf_r.mwavefronts = mwavefronts_r + wf_base_idx;
    wf_r.iwavefronts = iwavefronts_r + wf_base_idx;
    wf_r.dwavefronts = dwavefronts_r + wf_base_idx;

    for (int dir = 0; dir <= 1; dir++) {
        wf_components_t *wf = (dir == 0) ? &wf_f : &wf_r;
        int32_t *matrix_m = (dir == 0) ? matrix_wf_m_f : matrix_wf_m_r;
        int32_t *matrix_i = (dir == 0) ? matrix_wf_i_f : matrix_wf_i_r;
        int32_t *matrix_d = (dir == 0) ? matrix_wf_d_f : matrix_wf_d_r;

        wf->mwavefronts[0].offsets = matrix_m + alignment_id * total_offsets_size + 0 * wf_length + wf_length / 2;
        wf->iwavefronts[0].offsets = matrix_i + alignment_id * total_offsets_size + 0 * wf_length + wf_length / 2;
        wf->dwavefronts[0].offsets = matrix_d + alignment_id * total_offsets_size + 0 * wf_length + wf_length / 2;

        wf->mwavefronts[0].null = false;
        wf->mwavefronts[0].lo = -1;
        wf->mwavefronts[0].hi = 1;
        wf->mwavefronts[0].offsets[0] = 0;
        wf->mwavefronts[0].wf_elements_init_min = 0;
        wf->mwavefronts[0].wf_elements_init_max = 0;

        wf->iwavefronts[0].null = true;
        wf->dwavefronts[0].null = true;

        wf->wavefront_null.null = true;
        wf->wavefront_null.lo = 1;
        wf->wavefront_null.hi = -1;
        wf->wavefront_null.offsets = offsets_g + wf_length / 2;
    }

    int max_antidiag = text_len + pattern_len - 1;
    int score_f = 0, score_r = 0;
    int forward_max_ak = 0, reverse_max_ak = 0;
    int breakpoint_score = INT_MAX;
    bool finish = false;
    int alignment_k = text_len - pattern_len;

    extend_max(&finish, score_f, &forward_max_ak, &wf_f, max_score_scope, alignment_k, text_len, pattern_len);
    extend_max(&finish, score_r, &reverse_max_ak, &wf_r, max_score_scope, alignment_k, text_len, pattern_len);
    if (finish) return;

    int max_ak = 0;
    bool last_wf_forward = false;

    while (forward_max_ak + reverse_max_ak < max_antidiag) {
        nextWF(&score_f, &wf_f, true, max_score_scope, text_len, pattern_len, matrix_wf_m_f, matrix_wf_i_f, matrix_wf_d_f);
        extend_max(&finish, score_f, &max_ak, &wf_f, max_score_scope, alignment_k, text_len, pattern_len);
        forward_max_ak = max(forward_max_ak, max_ak);
        last_wf_forward = true;

        if (forward_max_ak + reverse_max_ak >= max_antidiag) break;

        nextWF(&score_r, &wf_r, false, max_score_scope, text_len, pattern_len, matrix_wf_m_r, matrix_wf_i_r, matrix_wf_d_r);
        extend_max(&finish, score_r, &max_ak, &wf_r, max_score_scope, alignment_k, text_len, pattern_len);
        reverse_max_ak = max(reverse_max_ak, max_ak);
        last_wf_forward = false;
    }

    int min_score_f, min_score_r;
    while (true) {
        if (last_wf_forward) {
            min_score_r = max(0, score_r - (max_score_scope - 1));
            if (score_f + min_score_r - penalty_gap_open >= breakpoint_score) break;
            overlap(score_f, &wf_f, score_r, &wf_r, max_score_scope, &breakpoint_score, text_len, pattern_len);
            nextWF(&score_r, &wf_r, true, max_score_scope, text_len, pattern_len, matrix_wf_m_r, matrix_wf_i_r, matrix_wf_d_r);
            extend(&finish, score_r, &wf_r, max_score_scope, alignment_k, text_len, pattern_len);
        }

        min_score_f = max(0, score_f - (max_score_scope - 1));
        if (min_score_f + score_r - penalty_gap_open >= breakpoint_score) break;
        overlap(score_r, &wf_r, score_f, &wf_f, max_score_scope, &breakpoint_score, text_len, pattern_len);
        nextWF(&score_f, &wf_f, false, max_score_scope, text_len, pattern_len, matrix_wf_m_f, matrix_wf_i_f, matrix_wf_d_f);
        extend(&finish, score_f, &wf_f, max_score_scope, alignment_k, text_len, pattern_len);

        if (score_r + score_f >= max_alignment_steps) break;
        last_wf_forward = true;
    }

    breakpoint_score = -breakpoint_score;
    breakpoint_score_g[alignment_id] = breakpoint_score;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_csv>\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        perror("File open error");
        return 1;
    }

    FILE *csv_file = fopen(argv[2], "w");
    if (!csv_file) {
        perror("CSV file open error");
        fclose(fp);
        return 1;
    }

    int num_alignments, pattern_len, text_len;
    if (fscanf(fp, "%d %d %d\n", &num_alignments, &pattern_len, &text_len) != 3) {
        printf("Error reading header.\n");
        fclose(fp);
        fclose(csv_file);
        return 1;
    }

    int *gpu_scores = (int *)malloc(sizeof(int) * num_alignments);
    auto total_start = std::chrono::high_resolution_clock::now();

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t matrix_size_per_alignment = sizeof(int32_t) * num_wavefronts * wf_length;
    size_t estimated_required_mem_per_alignment = matrix_size_per_alignment * 12;
    size_t max_batch_size = 100;
    if (max_batch_size > 10000) max_batch_size = 10000;
    if (max_batch_size < 1) {
        fprintf(stderr, "Not enough memory for even a single alignment.\n");
        free(gpu_scores);
        fclose(fp);
        fclose(csv_file);
        return 1;
    }

    printf("GPU Memory - Free: %zu bytes\n", free_mem);

    int correct_alignments = 0; 
    char temp_buffer[1024];
    for (int offset = 0; offset < num_alignments; offset += max_batch_size) {
        int current_batch_size = (offset + max_batch_size <= num_alignments)
                                 ? max_batch_size
                                 : (num_alignments - offset);

        auto batch_start = std::chrono::high_resolution_clock::now(); 

        int *pattern_lengths = (int *)malloc(sizeof(int) * current_batch_size);
        int *text_lengths = (int *)malloc(sizeof(int) * current_batch_size);
        int *pattern_offsets = (int *)malloc(sizeof(int) * current_batch_size);
        int *text_offsets = (int *)malloc(sizeof(int) * current_batch_size);

        int total_pattern_len = pattern_len * current_batch_size;
        int total_text_len = text_len * current_batch_size;

        char *pattern_concat = (char *)malloc(total_pattern_len);
        char *text_concat = (char *)malloc(total_text_len);
        char *pattern_r_concat = (char *)malloc(total_pattern_len);
        char *text_r_concat = (char *)malloc(total_text_len);

        bool read_error = false;

        for (int i = 0; i < current_batch_size; ++i) {
            pattern_lengths[i] = pattern_len;
            text_lengths[i] = text_len;
            pattern_offsets[i] = i * pattern_len;
            text_offsets[i] = i * text_len;

            if (!fgets(temp_buffer, sizeof(temp_buffer), fp)) {
                printf("Error reading pattern for alignment %d\n", offset + i);
                read_error = true;
                break;
            }
            strncpy(pattern_concat + pattern_offsets[i], temp_buffer, pattern_len);

            if (!fgets(temp_buffer, sizeof(temp_buffer), fp)) {
                printf("Error reading text for alignment %d\n", offset + i);
                read_error = true;
                break;
            }
            strncpy(text_concat + text_offsets[i], temp_buffer, text_len);

            for (int j = 0; j < pattern_len; ++j)
                pattern_r_concat[pattern_offsets[i] + j] = pattern_concat[pattern_offsets[i] + pattern_len - 1 - j];
            for (int j = 0; j < text_len; ++j)
                text_r_concat[text_offsets[i] + j] = text_concat[text_offsets[i] + text_len - 1 - j];
        }

        if (read_error) {
            free(pattern_concat); 
            free(text_concat);
            free(pattern_r_concat); 
            free(text_r_concat);
            free(pattern_lengths); 
            free(text_lengths);
            free(pattern_offsets); 
            free(text_offsets);
            free(gpu_scores);
            fclose(fp);
            fclose(csv_file);
            return 1;
        }

        char *d_pattern_concat, *d_text_concat, *d_pattern_r_concat, *d_text_r_concat;
        int *d_pattern_lengths, *d_text_lengths, *d_pattern_offsets, *d_text_offsets, *d_breakpoint_score;
        int32_t *d_offsets;
        size_t matrix_size = matrix_size_per_alignment * current_batch_size;

        wf_t *d_mwavefronts_f, *d_iwavefronts_f, *d_dwavefronts_f;
        wf_t *d_mwavefronts_r, *d_iwavefronts_r, *d_dwavefronts_r;
        int32_t *d_matrix_wf_m_f, *d_matrix_wf_i_f, *d_matrix_wf_d_f;
        int32_t *d_matrix_wf_m_r, *d_matrix_wf_i_r, *d_matrix_wf_d_r;

        CHECK(cudaMalloc(&d_pattern_concat, total_pattern_len));
        CHECK(cudaMalloc(&d_text_concat, total_text_len));
        CHECK(cudaMalloc(&d_pattern_r_concat, total_pattern_len));
        CHECK(cudaMalloc(&d_text_r_concat, total_text_len));
        CHECK(cudaMalloc(&d_pattern_lengths, sizeof(int) * current_batch_size));
        CHECK(cudaMalloc(&d_text_lengths, sizeof(int) * current_batch_size));
        CHECK(cudaMalloc(&d_pattern_offsets, sizeof(int) * current_batch_size));
        CHECK(cudaMalloc(&d_text_offsets, sizeof(int) * current_batch_size));
        CHECK(cudaMalloc(&d_breakpoint_score, sizeof(int) * current_batch_size));
        CHECK(cudaMalloc(&d_offsets, sizeof(int32_t) * wf_length));

        CHECK(cudaMalloc(&d_mwavefronts_f, matrix_size));
        CHECK(cudaMalloc(&d_iwavefronts_f, matrix_size));
        CHECK(cudaMalloc(&d_dwavefronts_f, matrix_size));
        CHECK(cudaMalloc(&d_mwavefronts_r, matrix_size));
        CHECK(cudaMalloc(&d_iwavefronts_r, matrix_size));
        CHECK(cudaMalloc(&d_dwavefronts_r, matrix_size));
        CHECK(cudaMalloc(&d_matrix_wf_m_f, matrix_size));
        CHECK(cudaMalloc(&d_matrix_wf_i_f, matrix_size));
        CHECK(cudaMalloc(&d_matrix_wf_d_f, matrix_size));
        CHECK(cudaMalloc(&d_matrix_wf_m_r, matrix_size));
        CHECK(cudaMalloc(&d_matrix_wf_i_r, matrix_size));
        CHECK(cudaMalloc(&d_matrix_wf_d_r, matrix_size));

        CHECK(cudaMemcpy(d_pattern_concat, pattern_concat, total_pattern_len, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_text_concat, text_concat, total_text_len, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_pattern_r_concat, pattern_r_concat, total_pattern_len, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_text_r_concat, text_r_concat, total_text_len, cudaMemcpyHostToDevice)); // Corretto qui
        CHECK(cudaMemcpy(d_pattern_lengths, pattern_lengths, sizeof(int) * current_batch_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_text_lengths, text_lengths, sizeof(int) * current_batch_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_pattern_offsets, pattern_offsets, sizeof(int) * current_batch_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_text_offsets, text_offsets, sizeof(int) * current_batch_size, cudaMemcpyHostToDevice));

        int32_t *temp_offsets = (int32_t *)malloc(sizeof(int32_t) * wf_length);
        for (int i = 0; i < wf_length; ++i) temp_offsets[i] = OFFSET_NULL;
        CHECK(cudaMemcpy(d_offsets, temp_offsets, sizeof(int32_t) * wf_length, cudaMemcpyHostToDevice));
        free(temp_offsets);

        int max_score_scope = max(penalty_gap_open + penalty_gap_ext, penalty_mismatch) + 1;
        int hi = max_score_scope + 1;
        int lo = -max_score_scope - 1;

        dim3 blocks(current_batch_size);
        dim3 threads(NUM_THREADS);
        biWFA_kernel<<<blocks, threads>>>(d_pattern_concat, d_text_concat, d_pattern_r_concat, d_text_r_concat,
                                          d_pattern_lengths, d_text_lengths, d_pattern_offsets, d_text_offsets,
                                          d_breakpoint_score,
                                          d_mwavefronts_f, d_iwavefronts_f, d_dwavefronts_f,
                                          d_mwavefronts_r, d_iwavefronts_r, d_dwavefronts_r,
                                          lo, hi, d_offsets, max_score_scope,
                                          d_matrix_wf_m_f, d_matrix_wf_i_f, d_matrix_wf_d_f,
                                          d_matrix_wf_m_r, d_matrix_wf_i_r, d_matrix_wf_d_r);

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        int *breakpoint_score = (int *)malloc(sizeof(int) * current_batch_size);
        CHECK(cudaMemcpy(breakpoint_score, d_breakpoint_score, sizeof(int) * current_batch_size, cudaMemcpyDeviceToHost));

        auto batch_end = std::chrono::high_resolution_clock::now(); 
        std::chrono::duration<double> batch_duration = batch_end - batch_start;
        double alignments_per_second = current_batch_size / batch_duration.count();

        printf("\nResults for batch %d-%d:\n", offset, offset + current_batch_size - 1);
        printf("Batch time: %.6f seconds | Alignments per second: %.2f\n", batch_duration.count(), alignments_per_second);

        wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
        attributes.distance_metric = gap_affine;
        attributes.affine_penalties.mismatch = penalty_mismatch;
        attributes.affine_penalties.gap_opening = penalty_gap_open;
        attributes.affine_penalties.gap_extension = penalty_gap_ext;
        wavefront_aligner_t *wf_aligner = wavefront_aligner_new(&attributes);

        for (int check_idx = 0; check_idx < current_batch_size; ++check_idx) {
            const char *pattern = pattern_concat + pattern_offsets[check_idx];
            const char *text = text_concat + text_offsets[check_idx];
            wavefront_align(wf_aligner, pattern, pattern_len, text, text_len);
            int cpu_score = wf_aligner->cigar->score;
            int gpu_score = breakpoint_score[check_idx];

            printf("Alignment %d:\n", offset + check_idx);
            printf("CPU Score: %d\n", cpu_score);
            printf("GPU Score: %d\n", gpu_score);
            printf("Match: %s\n\n", (cpu_score == gpu_score) ? "YES" : "NO");

            if (cpu_score == gpu_score) {
                correct_alignments++; 
            }
        }

        wavefront_aligner_delete(wf_aligner);

        free(breakpoint_score);
        cudaFree(d_pattern_concat); 
        cudaFree(d_text_concat);
        cudaFree(d_pattern_r_concat); 
        cudaFree(d_text_r_concat);
        cudaFree(d_pattern_lengths); 
        cudaFree(d_text_lengths);
        cudaFree(d_pattern_offsets); 
        cudaFree(d_text_offsets);
        cudaFree(d_breakpoint_score); 
        cudaFree(d_offsets);
        cudaFree(d_mwavefronts_f); 
        cudaFree(d_iwavefronts_f); 
        cudaFree(d_dwavefronts_f);
        cudaFree(d_mwavefronts_r); 
        cudaFree(d_iwavefronts_r); 
        cudaFree(d_dwavefronts_r);
        cudaFree(d_matrix_wf_m_f);
        cudaFree(d_matrix_wf_i_f); 
        cudaFree(d_matrix_wf_d_f);
        cudaFree(d_matrix_wf_m_r); 
        cudaFree(d_matrix_wf_i_r); 
        cudaFree(d_matrix_wf_d_r);
        free(pattern_concat); 
        free(text_concat);
        free(pattern_r_concat); 
        free(text_r_concat);
        free(pattern_lengths); 
        free(text_lengths);
        free(pattern_offsets); 
        free(text_offsets);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;

    double alignments_per_second = num_alignments / total_duration.count();

    printf("\nTotal correct alignments: %d\n", correct_alignments);
    printf("Total execution time: %.3f seconds\n", total_duration.count());
    printf("Alignments per second: %.3f\n", alignments_per_second);

    fprintf(csv_file, "Total correct alignments,%d\n", correct_alignments);
    fprintf(csv_file, "Total execution time (seconds),%.3f\n", total_duration.count());
    fprintf(csv_file, "Alignments per second,%.3f\n", alignments_per_second);

    free(gpu_scores);
    fclose(fp);
    fclose(csv_file);
    return 0;
}