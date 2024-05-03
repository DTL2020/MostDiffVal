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

	if (bits == 8)
	{
#pragma omp parallel for num_threads(threads)
		for (int y = 0; y < height; y++)
		{
			pixel_t* l_dstp = dstp + y * dst_pitch;
			pixel_t* l_srcp_ref = srcp_ref + y * src_pitch_ref;
			pixel_t* l_srcp_c1 = srcp_c1 + y * src_pitch_c1;
			pixel_t* l_srcp_c2 = srcp_c2 + y * src_pitch_c2;

			/*					dstp += dst_pitch;
					srcp_ref += src_pitch_ref;
					srcp_c1 += src_pitch_c1;
					srcp_c2 += src_pitch_c2;*/


/*			if (cpuFlags & CPUF_AVX512F) // use AVX512
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
				else*/
					if (cpuFlags & CPUF_SSE4_1) // use SSE up to 4.1
					{
						uint8_t* p_src_ref = (uint8_t*)l_srcp_ref;
						uint8_t* p_src_c1 = (uint8_t*)l_srcp_c1;
						uint8_t* p_src_c2 = (uint8_t*)l_srcp_c2;
						uint8_t* p_dst = (uint8_t*)l_dstp;
//						const int col32 = row_size - (row_size % 32); // use 16*2 128bit regs to load/store (is it good for current AVS+ row stride ?)
						const int col32 = src_pitch_ref - (src_pitch_ref % 32); // use 16*2 128bit regs to load/store (is it good for current AVS+ row stride ?)

						for (int64_t col = 0; col < col32; col += 32)
						{
							__m128i ref_0_15 = _mm_load_si128((__m128i*)p_src_ref); // hope always aligned to 128
							__m128i ref_16_31 = _mm_load_si128((__m128i*)(p_src_ref + 16));

							__m128i c1_0_15 = _mm_load_si128((__m128i*)p_src_c1); 
							__m128i c1_16_31 = _mm_load_si128((__m128i*)(p_src_c1 + 16));

							__m128i c2_0_15 = _mm_load_si128((__m128i*)p_src_c2); 
							__m128i c2_16_31 = _mm_load_si128((__m128i*)(p_src_c2 + 16));

							__m128i dif_c1_0_15 = _mm_sub_epi8(ref_0_15, c1_0_15);
							__m128i dif_c1_16_31 = _mm_sub_epi8(ref_16_31, c1_16_31);

							__m128i dif_c2_0_15 = _mm_sub_epi8(ref_0_15, c2_0_15);
							__m128i dif_c2_16_31 = _mm_sub_epi8(ref_16_31, c2_16_31);

							__m128i abs_dif_c1_0_15 = _mm_abs_epi8(dif_c1_0_15);//SSSE3
							__m128i abs_dif_c1_16_31 = _mm_abs_epi8(dif_c1_16_31);//SSSE3

							__m128i abs_dif_c2_0_15 = _mm_abs_epi8(dif_c2_0_15);//SSSE3
							__m128i abs_dif_c2_16_31 = _mm_abs_epi8(dif_c2_16_31);//SSSE3

							__m128i mask_0_15 = _mm_cmpgt_epi8(abs_dif_c1_0_15, abs_dif_c2_0_15);
							__m128i mask_16_31 = _mm_cmpgt_epi8(abs_dif_c1_16_31, abs_dif_c2_16_31);

							__m128i res_0_15 = _mm_blendv_epi8(c2_0_15, c1_0_15, mask_0_15);// SSE 4.1
							__m128i res_16_31 = _mm_blendv_epi8(c2_16_31, c1_16_31, mask_16_31);

							_mm_store_si128((__m128i*)p_dst, res_0_15);
							_mm_store_si128((__m128i*)(p_dst + 16), res_16_31);

							p_src_ref += 32; // in bytes
							p_src_c1 += 32; // in bytes
							p_src_c2 += 32; // in bytes
							p_dst += 32;
						}

						// last cols
						for (int64_t col = col32; col < row_size; ++col)
						{
							int iDiffC1 = std::abs(l_srcp_ref[col] - l_srcp_c1[col]);
							int iDiffC2 = std::abs(l_srcp_ref[col] - l_srcp_c2[col]);
							if (iDiffC1 > iDiffC2)
								l_dstp[col] = l_srcp_c1[col];
							else
								l_dstp[col] = l_srcp_c2[col];
						}
					}
					else // C-reference
						for (int x = 0; x < row_size; x++)
						{
							int iDiffC1 = std::abs(l_srcp_ref[x] - l_srcp_c1[x]);
							int iDiffC2 = std::abs(l_srcp_ref[x] - l_srcp_c2[x]);
							if (iDiffC1 > iDiffC2)
								l_dstp[x] = l_srcp_c1[x];
							else
								l_dstp[x] = l_srcp_c2[x];
						}
		}
	}
	else if (bits == 16)
	{
#pragma omp parallel for num_threads(threads)
		for (int y = 0; y < height; y++)
		{
			pixel_t* l_dstp = dstp + y * dst_pitch;
			pixel_t* l_srcp_ref = srcp_ref + y * src_pitch_ref;
			pixel_t* l_srcp_c1 = srcp_c1 + y * src_pitch_c1;
			pixel_t* l_srcp_c2 = srcp_c2 + y * src_pitch_c2;

			/*					dstp += dst_pitch;
					srcp_ref += src_pitch_ref;
					srcp_c1 += src_pitch_c1;
					srcp_c2 += src_pitch_c2;*/


					/*			if (cpuFlags & CPUF_AVX512F) // use AVX512
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
									else*/
			if (cpuFlags & CPUF_SSE4_1) // use SSE up to 4.1
			{
				uint8_t* p_src_ref = (uint8_t*)l_srcp_ref;
				uint8_t* p_src_c1 = (uint8_t*)l_srcp_c1;
				uint8_t* p_src_c2 = (uint8_t*)l_srcp_c2;
				uint8_t* p_dst = (uint8_t*)l_dstp;
				//						const int col32 = row_size - (row_size % 32); // use 16*2 128bit regs to load/store (is it good for current AVS+ row stride ?)
				const int col16 = src_pitch_ref - (src_pitch_ref % 16); // use 8*2 128bit regs to load/store (is it good for current AVS+ row stride ?)

				for (int64_t col = 0; col < col16; col += 16)
				{
					__m128i ref_0_7 = _mm_load_si128((__m128i*)p_src_ref); // hope always aligned to 128
					__m128i ref_8_15 = _mm_load_si128((__m128i*)(p_src_ref + 16));

					__m128i c1_0_7 = _mm_load_si128((__m128i*)p_src_c1);
					__m128i c1_8_15 = _mm_load_si128((__m128i*)(p_src_c1 + 16));

					__m128i c2_0_7 = _mm_load_si128((__m128i*)p_src_c2);
					__m128i c2_8_15 = _mm_load_si128((__m128i*)(p_src_c2 + 16));

					__m128i dif_c1_0_7 = _mm_sub_epi16(ref_0_7, c1_0_7);
					__m128i dif_c1_8_15 = _mm_sub_epi16(ref_8_15, c1_8_15);

					__m128i dif_c2_0_7 = _mm_sub_epi16(ref_0_7, c2_0_7);
					__m128i dif_c2_8_15 = _mm_sub_epi16(ref_8_15, c2_8_15);

					__m128i abs_dif_c1_0_7 = _mm_abs_epi16(dif_c1_0_7);//SSSE3
					__m128i abs_dif_c1_8_15 = _mm_abs_epi16(dif_c1_8_15);//SSSE3

					__m128i abs_dif_c2_0_7 = _mm_abs_epi16(dif_c2_0_7);//SSSE3
					__m128i abs_dif_c2_8_15 = _mm_abs_epi16(dif_c2_8_15);//SSSE3

					__m128i mask_0_7 = _mm_cmpgt_epi16(abs_dif_c1_0_7, abs_dif_c2_0_7);
					__m128i mask_8_15 = _mm_cmpgt_epi16(abs_dif_c1_8_15, abs_dif_c2_8_15);

					__m128i res_0_7 = _mm_blendv_epi8(c2_0_7, c1_0_7, mask_0_7);// SSE 4.1
					__m128i res_8_15 = _mm_blendv_epi8(c2_8_15, c1_8_15, mask_8_15);

					_mm_store_si128((__m128i*)p_dst, res_0_7);
					_mm_store_si128((__m128i*)(p_dst + 16), res_8_15);

					p_src_ref += 32; // in bytes
					p_src_c1 += 32; // in bytes
					p_src_c2 += 32; // in bytes
					p_dst += 32;
				}

				// last cols
				for (int64_t col = col16; col < row_size; ++col)
				{
					int iDiffC1 = std::abs(l_srcp_ref[col] - l_srcp_c1[col]);
					int iDiffC2 = std::abs(l_srcp_ref[col] - l_srcp_c2[col]);
					if (iDiffC1 > iDiffC2)
						l_dstp[col] = l_srcp_c1[col];
					else
						l_dstp[col] = l_srcp_c2[col];
				}
			}
			else // C-reference
				for (int x = 0; x < row_size; x++)
				{
					int iDiffC1 = std::abs(l_srcp_ref[x] - l_srcp_c1[x]);
					int iDiffC2 = std::abs(l_srcp_ref[x] - l_srcp_c2[x]);
					if (iDiffC1 > iDiffC2)
						l_dstp[x] = l_srcp_c1[x];
					else
						l_dstp[x] = l_srcp_c2[x];
				}
		}
	}
	else // bits 32
	{
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