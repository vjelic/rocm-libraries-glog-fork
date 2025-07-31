// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "hipfft/hipfftw.h"
#include "../../../shared/arithmetic.h"
#include "../../../shared/gpubuf.h"
#include "../../../shared/precision_type.h"
#include "../../../shared/ptrdiff.h"
#include "rocfft/rocfft.h"
#include "stride.h"
#include <memory>
#include <vector>

enum class fftw_transform_type
{
    COMPLEX,
    REAL,
};

struct fftw_plan_internal
{
    fftw_plan_internal() = default;
    ~fftw_plan_internal()
    {
        if(info)
        {
            rocfft_execution_info_destroy(info);
            info = nullptr;
        }
        if(desc)
        {
            rocfft_plan_description_destroy(desc);
            desc = nullptr;
        }
        if(plan)
        {
            rocfft_plan_destroy(plan);
            plan = nullptr;
        }
    }

    // disallow copies
    fftw_plan_internal(const fftw_plan_internal&) = delete;
    fftw_plan_internal& operator=(const fftw_plan_internal&) = delete;

    rocfft_plan             plan = nullptr;
    rocfft_plan_description desc = nullptr;
    rocfft_execution_info   info = nullptr;
    // Non-owning pointers to user data
    void* in  = nullptr;
    void* out = nullptr;

    // Sizes of the buffers so we know how much to copy
    size_t in_bytes  = 0;
    size_t out_bytes = 0;

    bool create(fftw_transform_type type,
                rocfft_precision    precision,
                int                 rank,
                int*                lengths_rm,
                int                 sign,
                void*               user_in,
                void*               user_out,
                int                 batch,
                const int*          inembed,
                int                 istride,
                int                 idist,
                const int*          onembed,
                int                 ostride,
                int                 odist)
    {
        // magic static to handle rocfft setup/cleanup
        struct rocfft_initializer
        {
            rocfft_initializer()
            {
                rocfft_setup();
            }
            ~rocfft_initializer()
            {
                rocfft_cleanup();
            }
        };
        static rocfft_initializer init;

        in  = user_in;
        out = user_out;

        // Change row-major to col-major
        std::vector<size_t> lengths_cm(rank);
        std::copy(std::reverse_iterator(lengths_rm + rank),
                  std::reverse_iterator(lengths_rm),
                  lengths_cm.begin());

        // Create plan description if necessary

        // FIXME: unify this with what the rest of the library is
        // doing, particularly with default strides
        if(inembed != nullptr || onembed != nullptr)
        {
            if(rocfft_plan_description_create(&desc) != rocfft_status_success)
                return false;

            auto array_types = get_rocfft_array_types(type, sign);
            auto in_strides  = embed_to_rocfft_stride(rank, lengths_rm, istride, inembed);
            auto out_strides = embed_to_rocfft_stride(rank, lengths_rm, ostride, onembed);

            if(rocfft_plan_description_set_data_layout(desc,
                                                       array_types.first,
                                                       array_types.second,
                                                       nullptr,
                                                       nullptr,
                                                       rank,
                                                       in_strides.data(),
                                                       idist,
                                                       rank,
                                                       out_strides.data(),
                                                       odist)
               != rocfft_status_success)
                return false;

            in_bytes  = compute_ptrdiff(lengths_cm, in_strides, batch, idist);
            out_bytes = compute_ptrdiff(lengths_cm, in_strides, batch, idist);
        }
        else
        {
            // compute default strides

            switch(type)
            {
            case fftw_transform_type::COMPLEX:
            {
                in_bytes = product(lengths_cm.begin(), lengths_cm.end()) * batch
                           * complex_type_size(precision);
                out_bytes = in_bytes;
                break;
            }
            case fftw_transform_type::REAL:
            {
                auto complex_length_cm = lengths_cm;
                complex_length_cm[0]   = complex_length_cm[0] / 2 + 1;
                auto real_bytes        = product(lengths_cm.begin(), lengths_cm.end()) * batch
                                  * real_type_size(precision);
                auto complex_bytes = product(complex_length_cm.begin(), complex_length_cm.end())
                                     * batch * complex_type_size(precision);
                if(sign == FFTW_FORWARD)
                {
                    in_bytes  = real_bytes;
                    out_bytes = complex_bytes;
                }
                else
                {
                    in_bytes  = complex_bytes;
                    out_bytes = real_bytes;
                }
                break;
            }
            }
        }

        if(rocfft_plan_create(&plan,
                              get_placement(user_in, user_out),
                              get_rocfft_transform_type(type, sign),
                              precision,
                              rank,
                              lengths_cm.data(),
                              batch,
                              desc)
           != rocfft_status_success)
            return false;
        return true;
    }

private:
    static rocfft_result_placement get_placement(void* in, void* out)
    {
        return in == out ? rocfft_placement_inplace : rocfft_placement_notinplace;
    }

    static rocfft_transform_type get_rocfft_transform_type(fftw_transform_type type, int sign)
    {
        if(type == fftw_transform_type::COMPLEX)
            return sign == FFTW_FORWARD ? rocfft_transform_type_complex_forward
                                        : rocfft_transform_type_complex_inverse;
        else
            return sign == FFTW_FORWARD ? rocfft_transform_type_real_forward
                                        : rocfft_transform_type_real_inverse;
    }

    // Return input/output array types as as std::pair
    static std::pair<rocfft_array_type, rocfft_array_type>
        get_rocfft_array_types(fftw_transform_type type, int sign)
    {
        std::pair<rocfft_array_type, rocfft_array_type> result;
        if(type == fftw_transform_type::COMPLEX)
        {
            result.first  = rocfft_array_type_complex_interleaved;
            result.second = rocfft_array_type_complex_interleaved;
        }
        else
        {
            if(sign == FFTW_FORWARD)
            {
                result.first  = rocfft_array_type_real;
                result.second = rocfft_array_type_hermitian_interleaved;
            }
            else
            {
                result.first  = rocfft_array_type_hermitian_interleaved;
                result.second = rocfft_array_type_real;
            }
        }
        return result;
    }
};

void* fftw_malloc(size_t n)
{
    return malloc(n);
}

void* fftwf_malloc(size_t n)
{
    return malloc(n);
}

void fftw_free(void* p)
{
    return free(p);
}
void fftwf_free(void* p)
{
    return free(p);
}

void fftw_destroy_plan(fftw_plan plan)
{
    auto internal_plan = static_cast<fftw_plan_internal*>(plan);
    delete internal_plan;
}
void fftwf_destroy_plan(fftwf_plan plan)
{
    auto internal_plan = static_cast<fftw_plan_internal*>(plan);
    delete internal_plan;
}

void fftw_cleanup() {}
void fftwf_cleanup() {}

template <typename Tplan>
void hipfftw_execute(const Tplan plan)
{
    auto internal_plan = static_cast<fftw_plan_internal*>(plan);

    void* in_exec  = nullptr;
    void* out_exec = nullptr;

    // copy input to device
    gpubuf_t in_device;
    (void)in_device.alloc(internal_plan->in_bytes);
    (void)hipMemcpy(
        in_device.data(), internal_plan->in, internal_plan->in_bytes, hipMemcpyHostToDevice);
    in_exec = in_device.data();

    // allocate output if not in-place
    gpubuf_t out_device;
    if(internal_plan->in != internal_plan->out)
    {
        (void)out_device.alloc(internal_plan->out_bytes);
        out_exec = out_device.data();
    }

    rocfft_execute(internal_plan->plan, &in_exec, &out_exec, internal_plan->info);

    // copy output to host
    (void)hipMemcpy(internal_plan->out,
                    internal_plan->in == internal_plan->out ? in_device.data() : out_device.data(),
                    internal_plan->out_bytes,
                    hipMemcpyDeviceToHost);
}

void fftw_execute(const fftwf_plan plan)
{
    hipfftw_execute(plan);
}

void fftwf_execute(const fftwf_plan plan)
{
    hipfftw_execute(plan);
}

template <rocfft_precision precision>
void* hipfftw_plan_dft_1d(int n, void* in, void* out, int sign, unsigned flags)
{
    auto internal_plan = std::make_unique<fftw_plan_internal>();
    if(!internal_plan->create(fftw_transform_type::COMPLEX,
                              precision,
                              1,
                              &n,
                              sign,
                              in,
                              out,
                              1,
                              nullptr,
                              0,
                              0,
                              nullptr,
                              0,
                              0))
        return nullptr;
    return internal_plan.release();
}

fftw_plan fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
{
    return hipfftw_plan_dft_1d<rocfft_precision_double>(n, in, out, sign, flags);
}

fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
{
    return hipfftw_plan_dft_1d<rocfft_precision_single>(n, in, out, sign, flags);
}

fftw_plan
    fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft_2d(
    int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_3d(
    int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft_3d(
    int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft(
    int rank, const int* n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft(
    int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c_1d(int n, double* in, fftwf_complex* out, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, double* in, fftwf_complex* out, unsigned flags)
{
    return nullptr;
}

fftw_plan
    fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftwf_complex* out, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_r2c(int rank, const int* n, double* in, fftwf_complex* out, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r_1d(int n, fftwf_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftw_plan
    fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, double* out, unsigned flags)
{
    return nullptr;
}

void   fftw_print_plan(const fftw_plan) {}
void   fftwf_print_plan(const fftwf_plan) {}
void   fftw_set_timelimit(double) {}
void   fftwf_set_timelimit(double) {}
double fftw_cost(const fftw_plan)
{
    return 0.0;
}
double fftwf_cost(const fftw_plan)
{
    return 0.0;
}
void fftw_flops(const fftw_plan, double*, double*, double*) {}
void fftwf_flops(const fftw_plan, double*, double*, double*) {}
