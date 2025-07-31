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

#include "../hipfftw_params.h"

#include "../../shared/hostbuf.h"
#include "../../shared/rocfft_against_fftw.h"

#include <fftw3.h>
#include <gtest/gtest.h>
#include <optional>

static hipfftw_funcs& get_hipfftw_funcs()
{
    static hipfftw_funcs funcs;
    return funcs;
}

static hipfftwf_funcs& get_hipfftwf_funcs()
{
    static hipfftwf_funcs funcs;
    return funcs;
}

template <typename Funcs>
void test_malloc_free(Funcs& funcs)
{
    auto ptr = static_cast<double*>(funcs.malloc(sizeof(double)));
    ASSERT_NE(ptr, nullptr);
    *ptr = 1.234;
    funcs.free(ptr);
}

TEST(hipfftwTest, malloc_free)
{
    auto& funcs = get_hipfftw_funcs();
    test_malloc_free(funcs);
}

TEST(hipfftwTest, malloc_free_f)
{
    auto& funcs = get_hipfftwf_funcs();
    test_malloc_free(funcs);
}

template <typename Funcs>
void test_utility(Funcs& funcs)
{
    // call utility functions - they need to exist but don't need to work
    funcs.print_plan(nullptr);
    funcs.set_timelimit(0.0);
    funcs.cost(nullptr);
    funcs.flops(nullptr, nullptr, nullptr, nullptr);
    funcs.cleanup();
}

TEST(hipfftwTest, utility)
{
    auto& funcs = get_hipfftw_funcs();
    test_utility(funcs);
}

TEST(hipfftwTest, utility_f)
{
    auto& funcs = get_hipfftwf_funcs();
    test_utility(funcs);
}

template <typename Funcs>
void test_plan_dft_1d(Funcs& funcs)
{
    for(auto sign : {FFTW_FORWARD, FFTW_BACKWARD})
    {
        for(auto placement : {fft_placement_inplace, fft_placement_notinplace})
        {
            const int len = 8192;

            // generate input for both reference FFT and hipfftw, and corresponding output buffers
            std::vector<hostbuf> ref_input(1);
            ref_input.back().alloc(
                var_size<size_t>(funcs.precision, fft_array_type_complex_interleaved) * len);
            set_input<typename Funcs::Tfloat>(ref_input,
                                              fft_input_random_generator_host,
                                              fft_array_type_complex_interleaved,
                                              {static_cast<size_t>(len)},
                                              {static_cast<size_t>(len)},
                                              {static_cast<size_t>(1)},
                                              static_cast<size_t>(len),
                                              static_cast<size_t>(1),
                                              0,
                                              1,
                                              {},
                                              static_cast<size_t>(0),
                                              0,
                                              static_cast<size_t>(1),
                                              0);
            std::vector<hostbuf> hip_input(1);
            hip_input.back() = ref_input.back().copy();

            std::vector<hostbuf> ref_output(1);
            std::vector<hostbuf> hip_output(1);
            if(placement == fft_placement_inplace)
            {
                ref_output.back()
                    = hostbuf::make_nonowned(ref_input.back().data(), ref_input.back().size());
                hip_output.back()
                    = hostbuf::make_nonowned(hip_input.back().data(), hip_input.back().size());
            }
            else
            {
                ref_output.back().alloc(ref_input.back().size());
                hip_output.back().alloc(hip_input.back().size());
            }

            // create FFTW and hipFFTW 1D plans
            auto ref_plan = funcs.ref_plan_dft_1d(
                len,
                static_cast<typename Funcs::Tcomplex*>(ref_input.back().data()),
                static_cast<typename Funcs::Tcomplex*>(ref_output.back().data()),
                sign,
                FFTW_ESTIMATE);
            auto hip_plan = funcs.plan_dft_1d(
                len,
                static_cast<typename Funcs::Tcomplex*>(hip_input.back().data()),
                static_cast<typename Funcs::Tcomplex*>(hip_output.back().data()),
                sign,
                FFTW_ESTIMATE);

            // execute transforms
            funcs.ref_execute(ref_plan);
            funcs.execute(hip_plan);

            auto ref_norm = norm(ref_output,
                                 static_cast<size_t>(len),
                                 1,
                                 funcs.precision,
                                 fft_array_type_complex_interleaved,
                                 static_cast<size_t>(1),
                                 0,
                                 {0});

            const double linf_cutoff = type_epsilon(funcs.precision) * ref_norm.l_inf * log(len);

            // compare results
            auto diff = distance(ref_output,
                                 hip_output,
                                 static_cast<size_t>(len),
                                 1,
                                 funcs.precision,
                                 fft_array_type_complex_interleaved,
                                 static_cast<size_t>(1),
                                 0,
                                 fft_array_type_complex_interleaved,
                                 static_cast<size_t>(1),
                                 0,
                                 nullptr,
                                 linf_cutoff,
                                 {0},
                                 {0});

            EXPECT_TRUE(diff.l_inf <= linf_cutoff);
            EXPECT_TRUE(diff.l_2 / ref_norm.l_2 < sqrt(log2(len)) * type_epsilon(funcs.precision));
        }
    }
}

TEST(hipfftwTest, plan_dft_1d)
{
    auto& funcs = get_hipfftw_funcs();
    test_plan_dft_1d(funcs);
}

TEST(hipfftwTest, plan_dft_1d_f)
{
    auto& funcs = get_hipfftwf_funcs();
    test_plan_dft_1d(funcs);
}
