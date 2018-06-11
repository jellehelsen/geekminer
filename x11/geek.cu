extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_shabal.h"

}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"
#include "x13/cuda_x13.h"

// to test gpu hash on a null buffer
#define NULLTEST 0

#include <stdio.h>
#include <memory.h>

static uint32_t *d_hash[MAX_GPUS];

// Geek CPU Hash
extern "C" void geekhash(void *output, const void *input)
{
	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
    #define hashA hash
    #define hashB hash+64

	memset(hash, 0, sizeof hash);

	// blake80-bmw512-echo512-shabal512-groestl512-cubehash512-keccak512-hamsi512-simd512

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_keccak512_context ctx_keccak;
	sph_cubehash512_context ctx_cubehash;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_shabal512_context ctx_shabal;


	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, hashA);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, hashA, 64);
	sph_bmw512_close(&ctx_bmw, hashB);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hashB, 64);
	sph_echo512_close(&ctx_echo, hashA);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hashA, 64);
	sph_shabal512_close(&ctx_shabal, hashB);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hashB, 64);
	sph_groestl512_close(&ctx_groestl, hashA);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, hashA, 64);
	sph_cubehash512_close(&ctx_cubehash, hashB);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, hashB, 64);
	sph_keccak512_close(&ctx_keccak, hashA);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, hashA, 64);
	sph_hamsi512_close(&ctx_hamsi, hashB);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hashB, 64);
	sph_simd512_close(&ctx_simd, hashA);

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "geek"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_geek(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 19 : 18;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x2;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA

		// blake80-bmw-echo-shabal-groestl-cubehash-keccak-hamsi-simd

		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x14_shabal512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, NULL, d_hash[thr_id]); order++;
		x13_hamsi512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]); order++;
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

#if NULLTEST
		uint32_t buf[8]; memset(buf, 0, sizeof buf);
		CUDA_SAFE_CALL(cudaMemcpy(buf, d_hash[thr_id], sizeof buf, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		applog_hash(buf);
#endif

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			geekhash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					geekhash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_geek(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
