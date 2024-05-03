#include <windows.h>
#include "include\avisynth.h"
#include <immintrin.h>
#include <cstdlib>

template<typename pixel_t>
void ProcessPlane(unsigned char* _srcp_ref, unsigned char* _srcp_c1, unsigned char* _srcp_c2,
	unsigned char* _dstp,
	int src_pitch_ref, int src_pitch_c1, int src_pitch_c2,
	int dst_pitch, int height, int row_size, int bits, int threads, int cpuFlags)
{
	pixel_t* dstp = reinterpret_cast<pixel_t*>(_dstp);
	pixel_t* srcp_ref = reinterpret_cast<pixel_t*>(_srcp_ref);
	pixel_t* srcp_c1 = reinterpret_cast<pixel_t*>(_srcp_c1);
	pixel_t* srcp_c2 = reinterpret_cast<pixel_t*>(_srcp_c2);

/*	if (bits == 32)
	{
#pragma omp parallel for num_threads(threads)
		for (int y = 0; y < height; y++)
		{
			pixel_t* l_dstp = dstp + y * dst_pitch;
			pixel_t* l_srcp = srcp + y * src_pitch;

			if (cpuFlags & CPUF_AVX512F) // use AVX512
			{
				float* pf_src = (float*)l_srcp;
				float* pf_dst = (float*)l_dstp;
				const int col64 = row_size - (row_size % 64); // use 4*16 512bit regs to load/store
				__m512 zmm_fone = _mm512_set1_ps(1.0f);

				for (int64_t col = 0; col < col64; col += 64)
				{
					__m512 zmm0 = _mm512_loadu_ps(pf_src); // better align start addr with pre-conversion of 32(64?)-bytes aligned (if exist) and use load_ps
					__m512 zmm1 = _mm512_loadu_ps(pf_src + 16);
					__m512 zmm2 = _mm512_loadu_ps(pf_src + 32);
					__m512 zmm3 = _mm512_loadu_ps(pf_src + 48);

					zmm0 = _mm512_sub_ps(zmm_fone, zmm0);
					zmm1 = _mm512_sub_ps(zmm_fone, zmm1);
					zmm2 = _mm512_sub_ps(zmm_fone, zmm2);
					zmm3 = _mm512_sub_ps(zmm_fone, zmm3);

					_mm512_storeu_ps(pf_dst, zmm0);
					_mm512_storeu_ps(pf_dst + 16, zmm1);
					_mm512_storeu_ps(pf_dst + 32, zmm2);
					_mm512_storeu_ps(pf_dst + 48, zmm3);

					pf_src += 64; // in floats
					pf_dst += 64;
				}

				// last cols
				for (int64_t col = col64; col < row_size; ++col)
				{
					*pf_dst = (pixel_t)(1.0f - *pf_src);
					pf_dst++;
					pf_src++;
				}
			}
			else
				if (cpuFlags & CPUF_AVX) // use AVX
				{
					float* pf_src = (float*)l_srcp;
					float* pf_dst = (float*)l_dstp;
					const int col32 = row_size - (row_size % 32); // use 4*8 256bit regs to load/store
					__m256 ymm_fone = _mm256_set1_ps(1.0f);

					for (int64_t col = 0; col < col32; col += 32)
					{
						__m256 ymm0 = _mm256_loadu_ps(pf_src); // better align start addr with pre-conversion of 32-bytes aligned (if exist) and use load_ps
						__m256 ymm1 = _mm256_loadu_ps(pf_src + 8);
						__m256 ymm2 = _mm256_loadu_ps(pf_src + 16);
						__m256 ymm3 = _mm256_loadu_ps(pf_src + 24);

						ymm0 = _mm256_sub_ps(ymm_fone, ymm0);
						ymm1 = _mm256_sub_ps(ymm_fone, ymm1);
						ymm2 = _mm256_sub_ps(ymm_fone, ymm2);
						ymm3 = _mm256_sub_ps(ymm_fone, ymm3);

						_mm256_storeu_ps(pf_dst, ymm0);
						_mm256_storeu_ps(pf_dst + 8, ymm1);
						_mm256_storeu_ps(pf_dst + 16, ymm2);
						_mm256_storeu_ps(pf_dst + 24, ymm3);

						pf_src += 32; // in floats
						pf_dst += 32;
					}

					// last cols
					for (int64_t col = col32; col < row_size; ++col)
					{
						*pf_dst = (pixel_t)(1.0f - *pf_src);
						pf_dst++;
						pf_src++;
					}
				}
				else
					if (cpuFlags & CPUF_SSE) // use SSE
					{
						float* pf_src = (float*)l_srcp;
						float* pf_dst = (float*)l_dstp;
						const int col16 = row_size - (row_size % 16); // use 4*4 128bit regs to load/store
						__m128 xmm_fone = _mm_set1_ps(1.0f);

						for (int64_t col = 0; col < col16; col += 16)
						{
							__m128 xmm0 = _mm_loadu_ps(pf_src); // better align start addr with pre-conversion of 16-bytes aligned (if exist) and use load_ps
							__m128 xmm1 = _mm_loadu_ps(pf_src + 4);
							__m128 xmm2 = _mm_loadu_ps(pf_src + 8);
							__m128 xmm3 = _mm_loadu_ps(pf_src + 12);

							xmm0 = _mm_sub_ps(xmm_fone, xmm0);
							xmm1 = _mm_sub_ps(xmm_fone, xmm1);
							xmm2 = _mm_sub_ps(xmm_fone, xmm2);
							xmm3 = _mm_sub_ps(xmm_fone, xmm3);

							_mm_storeu_ps(pf_dst, xmm0);
							_mm_storeu_ps(pf_dst + 4, xmm1);
							_mm_storeu_ps(pf_dst + 8, xmm2);
							_mm_storeu_ps(pf_dst + 12, xmm3);

							pf_src += 16; // in floats
							pf_dst += 16;
						}

						// last cols
						for (int64_t col = col16; col < row_size; ++col)
						{
							*pf_dst = (pixel_t)(1.0f - *pf_src);
							pf_dst++;
							pf_src++;
						}
					}
					else // C-reference
						for (int x = 0; x < row_size; x++)
						{
							l_dstp[x] = (pixel_t)(1.0f - l_srcp[x]);
						}
		}
	}
	else*/
	{
	/*	pixel_t* dstp = reinterpret_cast<pixel_t*>(_dstp);
	pixel_t* srcp_ref = reinterpret_cast<pixel_t*>(_srcp);
	pixel_t* srcp_c1 = reinterpret_cast<pixel_t*>(_srcp_c1);
	pixel_t* srcp_c2 = reinterpret_cast<pixel_t*>(_srcp_c2);
*/
	typedef typename std::conditional <sizeof(pixel_t) <= 2, int, float>::type working_t;

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < row_size; x++)
			{
				working_t wtDiffC1 = std::abs(srcp_ref[x] - srcp_c1[x]);
				working_t wtDiffC2 = std::abs(srcp_ref[x] - srcp_c2[x]);
				if (wtDiffC1 > wtDiffC2) 
					dstp[x] = srcp_c1[x];
				else
					dstp[x] = srcp_c2[x];
			}
			dstp += dst_pitch;
			srcp_ref += src_pitch_ref;
			srcp_c1 += src_pitch_c1;
			srcp_c2 += src_pitch_c2;
		}
	}
}

template void ProcessPlane<uint8_t>(unsigned char* _srcp_ref, unsigned char* _srcp_c1, unsigned char* _srcp_c2, 
	unsigned char* _dstp, 
	int src_pitch_ref, int src_pitch_c1, int src_pitch_c2, 
	int dst_pitch, int height, int row_size, int bits, int threads, int cpuFlags);
template void ProcessPlane<uint16_t>(unsigned char* _srcp_ref, unsigned char* _srcp_c1, unsigned char* _srcp_c2,
	unsigned char* _dstp,
	int src_pitch_ref, int src_pitch_c1, int src_pitch_c2,
	int dst_pitch, int height, int row_size, int bits, int threads, int cpuFlags);
template void ProcessPlane<float>(unsigned char* _srcp_ref, unsigned char* _srcp_c1, unsigned char* _srcp_c2,
	unsigned char* _dstp,
	int src_pitch_ref, int src_pitch_c1, int src_pitch_c2,
	int dst_pitch, int height, int row_size, int bits, int threads, int cpuFlags);

class MostDiffVal : public GenericVideoFilter
{
	int threads;
	int _cpuFlags;

	PClip child_c1;
	PClip child_c2;

public:
	MostDiffVal(PClip _child_ref, PClip _child_c1, PClip _child_c2, int threads_, IScriptEnvironment* env) : GenericVideoFilter(_child_ref), threads(threads_)
	{
		_cpuFlags = env->GetCPUFlags();
		if (_child_c1 != 0)
			child_c1 = _child_c1;
		else
			env->ThrowError("MostDiffVal: No second clip param provided");

		if (_child_c2 != 0)
			child_c2 = _child_c2;
		else
			env->ThrowError("MostDiffVal: No third clip param provided");
	}
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		PVideoFrame dst = env->NewVideoFrame(vi);
		PVideoFrame src_ref = child->GetFrame(n, env);
		PVideoFrame src_c1 = child_c1->GetFrame(n, env);
		PVideoFrame src_c2 = child_c2->GetFrame(n, env);

		auto srcp_ref_Y = src_ref->GetReadPtr(PLANAR_Y);
		auto srcp_c1_Y = src_c1->GetReadPtr(PLANAR_Y);
		auto srcp_c2_Y = src_c2->GetReadPtr(PLANAR_Y);

		auto srcp_ref_U = src_ref->GetReadPtr(PLANAR_U);
		auto srcp_c1_U = src_c1->GetReadPtr(PLANAR_U);
		auto srcp_c2_U = src_c2->GetReadPtr(PLANAR_U);

		auto srcp_ref_V = src_ref->GetReadPtr(PLANAR_V);
		auto srcp_c1_V = src_c1->GetReadPtr(PLANAR_V);
		auto srcp_c2_V = src_c2->GetReadPtr(PLANAR_V);

		auto dstp_Y = dst->GetWritePtr(PLANAR_Y);
		auto dstp_U = dst->GetWritePtr(PLANAR_U);
		auto dstp_V = dst->GetWritePtr(PLANAR_V);
		auto height_Y = src_ref->GetHeight(PLANAR_Y);
		auto height_U = src_ref->GetHeight(PLANAR_U);
		auto height_V = src_ref->GetHeight(PLANAR_V);

		auto row_size_ref_Y = src_ref->GetRowSize(PLANAR_Y) / vi.ComponentSize();
		auto row_size_c1_Y = src_c1->GetRowSize(PLANAR_Y) / vi.ComponentSize();
		auto row_size_c2_Y = src_c1->GetRowSize(PLANAR_Y) / vi.ComponentSize();

		auto row_size_ref_U = src_ref->GetRowSize(PLANAR_U) / vi.ComponentSize();
		auto row_size_c1_U = src_c1->GetRowSize(PLANAR_U) / vi.ComponentSize();
		auto row_size_c2_U = src_c1->GetRowSize(PLANAR_U) / vi.ComponentSize();

		auto row_size_ref_V = src_ref->GetRowSize(PLANAR_V) / vi.ComponentSize();
		auto row_size_c1_V = src_c1->GetRowSize(PLANAR_V) / vi.ComponentSize();
		auto row_size_c2_V = src_c1->GetRowSize(PLANAR_V) / vi.ComponentSize();

		auto src_ref_pitch_Y = src_ref->GetPitch(PLANAR_Y) / vi.ComponentSize();
		auto src_c1_pitch_Y = src_c1->GetPitch(PLANAR_Y) / vi.ComponentSize();
		auto src_c2_pitch_Y = src_c2->GetPitch(PLANAR_Y) / vi.ComponentSize();

		auto src_ref_pitch_U = src_ref->GetPitch(PLANAR_U) / vi.ComponentSize();
		auto src_c1_pitch_U = src_c1->GetPitch(PLANAR_U) / vi.ComponentSize();
		auto src_c2_pitch_U = src_c2->GetPitch(PLANAR_U) / vi.ComponentSize();

		auto src_ref_pitch_V = src_ref->GetPitch(PLANAR_V) / vi.ComponentSize();
		auto src_c1_pitch_V = src_c1->GetPitch(PLANAR_V) / vi.ComponentSize();
		auto src_c2_pitch_V = src_c2->GetPitch(PLANAR_V) / vi.ComponentSize();

		auto dst_pitch_Y = dst->GetPitch(PLANAR_Y) / vi.ComponentSize();
		auto dst_pitch_U = dst->GetPitch(PLANAR_U) / vi.ComponentSize();
		auto dst_pitch_V = dst->GetPitch(PLANAR_V) / vi.ComponentSize();

		if (vi.ComponentSize() == 1)
		{
			// Y
			ProcessPlane<uint8_t>((uint8_t*)srcp_ref_Y, (uint8_t*)srcp_c1_Y, (uint8_t*)srcp_c2_Y, dstp_Y, src_ref_pitch_Y, src_c1_pitch_Y, src_c2_pitch_Y, 
				dst_pitch_Y, height_Y, row_size_ref_Y, vi.BitsPerComponent(), threads, _cpuFlags);
			// U
			ProcessPlane<uint8_t>((uint8_t*)srcp_ref_U, (uint8_t*)srcp_c1_U, (uint8_t*)srcp_c2_U, dstp_U, src_ref_pitch_U, src_c1_pitch_U, src_c2_pitch_U,
				dst_pitch_U, height_U, row_size_ref_U, vi.BitsPerComponent(), threads, _cpuFlags);
			// V
			ProcessPlane<uint8_t>((uint8_t*)srcp_ref_V, (uint8_t*)srcp_c1_V, (uint8_t*)srcp_c2_V, dstp_V, src_ref_pitch_V, src_c1_pitch_V, src_c2_pitch_V,
				dst_pitch_V, height_V, row_size_ref_V, vi.BitsPerComponent(), threads, _cpuFlags);

		}
		if (vi.ComponentSize() == 2)
		{
			// Y
			ProcessPlane<uint16_t>((uint8_t*)srcp_ref_Y, (uint8_t*)srcp_c1_Y, (uint8_t*)srcp_c2_Y, dstp_Y, src_ref_pitch_Y, src_c1_pitch_Y, src_c2_pitch_Y,
				dst_pitch_Y, height_Y, row_size_ref_Y, vi.BitsPerComponent(), threads, _cpuFlags);
			// U
			ProcessPlane<uint16_t>((uint8_t*)srcp_ref_U, (uint8_t*)srcp_c1_U, (uint8_t*)srcp_c2_U, dstp_U, src_ref_pitch_U, src_c1_pitch_U, src_c2_pitch_U,
				dst_pitch_U, height_U, row_size_ref_U, vi.BitsPerComponent(), threads, _cpuFlags);
			// V
			ProcessPlane<uint16_t>((uint8_t*)srcp_ref_V, (uint8_t*)srcp_c1_V, (uint8_t*)srcp_c2_V, dstp_V, src_ref_pitch_V, src_c1_pitch_V, src_c2_pitch_V,
				dst_pitch_V, height_V, row_size_ref_V, vi.BitsPerComponent(), threads, _cpuFlags);

		}
		if (vi.ComponentSize() == 4)
		{
			// Y
			ProcessPlane<float>((uint8_t*)srcp_ref_Y, (uint8_t*)srcp_c1_Y, (uint8_t*)srcp_c2_Y, dstp_Y, src_ref_pitch_Y, src_c1_pitch_Y, src_c2_pitch_Y,
				dst_pitch_Y, height_Y, row_size_ref_Y, vi.BitsPerComponent(), threads, _cpuFlags);
			// U
			ProcessPlane<float>((uint8_t*)srcp_ref_U, (uint8_t*)srcp_c1_U, (uint8_t*)srcp_c2_U, dstp_U, src_ref_pitch_U, src_c1_pitch_U, src_c2_pitch_U,
				dst_pitch_U, height_U, row_size_ref_U, vi.BitsPerComponent(), threads, _cpuFlags);
			// V
			ProcessPlane<float>((uint8_t*)srcp_ref_V, (uint8_t*)srcp_c1_V, (uint8_t*)srcp_c2_V, dstp_V, src_ref_pitch_V, src_c1_pitch_V, src_c2_pitch_V,
				dst_pitch_V, height_V, row_size_ref_V, vi.BitsPerComponent(), threads, _cpuFlags);

		}
		return dst;
	}
};


AVSValue __cdecl Create_MostDiffVal(AVSValue args, void* user_data, IScriptEnvironment* env)
{
	return new MostDiffVal(args[0].AsClip(), args[1].AsClip(), args[2].AsClip(), args[3].AsInt(1), env);
}

const AVS_Linkage* AVS_linkage = 0;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
	AVS_linkage = vectors;
	env->AddFunction("MostDiffVal", "ccc[threads]i", Create_MostDiffVal, 0);
	return "MostDiffVal plugin";
}