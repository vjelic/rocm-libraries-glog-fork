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

#include "../../shared/accuracy_test.h"
#include "../../shared/hostbuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_against_fftw.h"
#include "../../shared/test_params.h"

#include <cstdint>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <numeric>

#ifdef WIN32
#include <windows.h>
#else
#include <cstdlib>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// test details
namespace
{
    // structure enabling verbose exception handler for hipfftw and
    // redirecting std::cerr to a runtime buffer throughout its lifetime
    struct hipfftw_exception_logger
    {
        bool              disable_logger_upon_destruction;
        std::stringstream buffer;
        std::streambuf*   original_cerr_rdbuf = nullptr;

    public:
        hipfftw_exception_logger()
            : disable_logger_upon_destruction(false)
            , original_cerr_rdbuf(std::cerr.rdbuf())
        {
            const char* hipfftw_log_trace = std::getenv("HIPFFTW_LOG_EXCEPTIONS");
            if(!hipfftw_log_trace || std::atoi(hipfftw_log_trace) <= 0)
            {
#ifdef WIN32
                disable_logger_upon_destruction
                    = SetEnvironmentVariable("HIPFFTW_LOG_EXCEPTIONS", "1") != 0;
#else
                disable_logger_upon_destruction
                    = setenv("HIPFFTW_LOG_EXCEPTIONS", "1", 1) == EXIT_SUCCESS;
#endif
            }
            std::cerr.rdbuf(buffer.rdbuf());
        }
        hipfftw_exception_logger(const hipfftw_exception_logger&) = delete;
        hipfftw_exception_logger(hipfftw_exception_logger&&)      = delete;
        hipfftw_exception_logger& operator=(const hipfftw_exception_logger&) = delete;
        hipfftw_exception_logger& operator=(hipfftw_exception_logger&&) = delete;
        ~hipfftw_exception_logger()
        {
            if(disable_logger_upon_destruction)
            {
#ifdef WIN32
                SetEnvironmentVariable("HIPFFTW_LOG_EXCEPTIONS", "0");
#else
                setenv("HIPFFTW_LOG_EXCEPTIONS", "0", 1);
#endif
            }
            // restore cerr to its original state
            std::cerr.rdbuf(original_cerr_rdbuf);
        }
        std::string get_log() const
        {
            return buffer.str();
        }
    };

    struct hipfftw_malloc_params
    {
        enum class alloc_api_type
        {
            none,
            real,
            complex
        };
        // bit mask to prevent allocation kind(s)
        enum alloc_kind_disabler : unsigned
        {
            none          = 0x0,
            pinned_host   = 0x1,
            pageable_host = 0x2
        };
        size_t              alloc_arg;
        alloc_api_type      alloc_api;
        fft_precision       prec;
        alloc_kind_disabler avoid_alloc_kind;

        std::string to_string() const
        {
            if(prec != fft_precision_single && prec != fft_precision_double)
                throw std::runtime_error("unknown precision");
            if(alloc_api != alloc_api_type::none && alloc_api != alloc_api_type::real
               && alloc_api != alloc_api_type::complex)
                throw std::runtime_error("unknown precision");
            if(avoid_alloc_kind & ~(none | pinned_host | pageable_host))
                throw std::runtime_error("unknown allocation kind to avoid");

            std::string ret;
            if(prec == fft_precision_single)
                ret += "fftwf_";
            else
                ret += "fftw_";

            if(alloc_api == alloc_api_type::none)
                ret += "malloc_";
            else if(alloc_api == alloc_api_type::real)
                ret += "alloc_real_";
            else
                ret += "alloc_complex_";
            ret += std::to_string(alloc_arg);
            if(avoid_alloc_kind & pinned_host)
                ret += "_avoiding_pinned_host";
            if(avoid_alloc_kind & pageable_host)
                ret += "_avoiding_pageable_host";
            return ret;
        }
    };

    // test suite for testing hipfftw allocation APIs
    class allocation : public ::testing::TestWithParam<hipfftw_malloc_params>
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}

    public:
        static std::string TestName(const testing::TestParamInfo<allocation::ParamType>& info)
        {
            return info.param.to_string();
        }
    };

    template <fft_precision prec>
    void malloc_write_read_free_test(const hipfftw_malloc_params& params)
    {
        if(params.prec != prec)
            GTEST_FAIL() << "wrong test specialization used for the desired test parameters";
        if(params.alloc_api != hipfftw_malloc_params::alloc_api_type::none
           && params.alloc_api != hipfftw_malloc_params::alloc_api_type::real
           && params.alloc_api != hipfftw_malloc_params::alloc_api_type::complex)
            GTEST_FAIL() << "unknown allocation api";

        std::ostringstream       gtest_info;
        bool                     test_skipped                     = false;
        bool                     test_failed                      = false;
        bool                     pinned_host_alloc_was_disabled   = false;
        bool                     pageable_host_alloc_was_disabled = false;
        const auto&              hipfftw_      = hipfftw_funcs<prec>::get_instance();
        void*                    hipfftw_alloc = nullptr;
        hipfftw_exception_logger exception_logger;
        struct allocation_test_to_be_skipped : std::runtime_error
        {
            using std::runtime_error::runtime_error;
        };
        struct allocation_test_failed : std::runtime_error
        {
            using std::runtime_error::runtime_error;
        };
        struct allocation_test_success
        {
        }; // used to cut execution short when applicable

        try
        {
            // enable allocation limits, if relevant
            if(params.avoid_alloc_kind & hipfftw_malloc_params::pinned_host)
            {
#ifdef WIN32
                pinned_host_alloc_was_disabled
                    = SetEnvironmentVariable("HIPFFTW_ALLOC_LIMIT_PINNED_HOST", "0") != 0;
#else
                pinned_host_alloc_was_disabled
                    = setenv("HIPFFTW_ALLOC_LIMIT_PINNED_HOST", "0", 1) == EXIT_SUCCESS;
#endif
                if(!pinned_host_alloc_was_disabled)
                {
                    throw allocation_test_to_be_skipped(
                        "failed to set environment variable disabling pinned host allocation");
                }
            }
            if(params.avoid_alloc_kind & hipfftw_malloc_params::pageable_host)
            {
#ifdef WIN32
                pageable_host_alloc_was_disabled
                    = SetEnvironmentVariable("HIPFFTW_ALLOC_LIMIT_PAGEABLE_HOST", "0") ! = 0;
#else
                pageable_host_alloc_was_disabled
                    = setenv("HIPFFTW_ALLOC_LIMIT_PAGEABLE_HOST", "0", 1) == EXIT_SUCCESS;
#endif
                if(!pageable_host_alloc_was_disabled)
                {
                    throw allocation_test_to_be_skipped(
                        "failed to set environment variable disabling pageable host allocation");
                }
            }
            // The test
            // - fills values in the allocated arrays as 0, 1, ..., max_elem, 0, 1, ..., max_elem, 0, 1, etc.
            //   (max_elem, max_elem-1, ..., 1, 0, max_elem, max_elem-1, ..., for imaginary values)
            // - reads the values thereafter and accumulates them as a sum of doubles
            // - checks the result.
            // --> the expected_result's must be exactly representable as double values
            const size_t max_elem = static_cast<size_t>(std::numeric_limits<uint8_t>::max());
            const size_t cycle_sz = max_elem + 1;
            const size_t max_representable_result = 1ULL << std::numeric_limits<double>::digits;
            auto         sum_of_integers          = [](size_t to, size_t from = 0) {
                if(from > to)
                    throw std::invalid_argument("invalid argument for sum_of_integers lambda");
                return (to + from) * (to - from + 1) / 2;
            };
            const size_t sum_of_cycles = (params.alloc_arg / cycle_sz) * sum_of_integers(max_elem);
            const size_t tail_sz       = params.alloc_arg % cycle_sz;
            const size_t expected_result_r
                = sum_of_cycles + (tail_sz > 0 ? sum_of_integers(tail_sz - 1) : 0);
            const size_t expected_result_i
                = sum_of_cycles
                  + (tail_sz > 0 ? sum_of_integers(max_elem, max_elem + 1 - tail_sz) : 0);
            if(expected_result_r > max_representable_result
               || (params.alloc_api == hipfftw_malloc_params::alloc_api_type::complex
                   && expected_result_i > max_representable_result))
                throw allocation_test_to_be_skipped("Test cannot reliably check for argument "
                                                    + std::to_string(params.alloc_arg));

            if(params.alloc_api == hipfftw_malloc_params::alloc_api_type::none)
                hipfftw_alloc = hipfftw_.malloc(params.alloc_arg);
            else if(params.alloc_api == hipfftw_malloc_params::alloc_api_type::real)
                hipfftw_alloc = hipfftw_.alloc_real(params.alloc_arg);
            else
                hipfftw_alloc = hipfftw_.alloc_complex(params.alloc_arg);

            // check that the allocation behaved as expected
            if((pinned_host_alloc_was_disabled && pageable_host_alloc_was_disabled)
               || params.alloc_arg == 0)
            {
                if(!hipfftw_alloc)
                    throw allocation_test_success();
                else
                    // no allocation should have happened
                    throw allocation_test_failed("allocation misbehaved for zero-size request or "
                                                 "fully-disabled hipfftw allocation");
            }
            if(!hipfftw_alloc)
                throw allocation_test_failed("allocation failed");
                // check that the host can write to the entire allocation
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t idx = 0; idx < params.alloc_arg; idx++)
            {
                uint8_t val = static_cast<uint8_t>(idx % cycle_sz);
                if(params.alloc_api
                   == hipfftw_malloc_params::alloc_api_type::none) // write as uint8_t
                    static_cast<uint8_t*>(hipfftw_alloc)[idx] = val;
                else if(params.alloc_api
                        == hipfftw_malloc_params::alloc_api_type::real) // write as float/double
                    static_cast<hipfftw_real_t<prec>*>(hipfftw_alloc)[idx] = val;
                else // write as complex value of req. precision
                {
                    static_cast<hipfftw_complex_t<prec>*>(hipfftw_alloc)[idx][0] = val;
                    static_cast<hipfftw_complex_t<prec>*>(hipfftw_alloc)[idx][1] = max_elem - val;
                }
            }
            // check that the host can read from the entire allocation
            double result[2] = {0, 0};
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : result)
#endif
            for(size_t idx = 0; idx < params.alloc_arg; idx++)
            {
                if(params.alloc_api == hipfftw_malloc_params::alloc_api_type::none)
                {
                    // read as uint8_t, accumulate as double
                    result[0] += static_cast<uint8_t*>(hipfftw_alloc)[idx];
                }
                else if(params.alloc_api == hipfftw_malloc_params::alloc_api_type::real)
                {
                    // read as float/double, accumulate as double
                    result[0] += static_cast<hipfftw_real_t<prec>*>(hipfftw_alloc)[idx];
                }
                else // write as complex value of req. precision
                {
                    // read as complex value of req. precision, accumulate as doubles
                    result[0] += static_cast<hipfftw_complex_t<prec>*>(hipfftw_alloc)[idx][0];
                    result[1] += static_cast<hipfftw_complex_t<prec>*>(hipfftw_alloc)[idx][1];
                }
            }
            // validity checks
            if(result[0] != expected_result_r)
                throw allocation_test_failed("incorrect result for accumulated real parts");
            if(params.alloc_api == hipfftw_malloc_params::alloc_api_type::complex
               && result[1] != expected_result_i)
                throw allocation_test_failed("incorrect result for accumulated imaginary parts");
        }
        catch(const allocation_test_success&)
        {
            // so far so good
        }
        catch(const hipfftw_undefined_function_ptr& e)
        {
            gtest_info << "undefined function pointers detected. Error info: " << e.what();
            test_skipped = true;
        }
        catch(const allocation_test_to_be_skipped& e)
        {
            gtest_info << e.what();
            test_skipped = true;
        }
        catch(const allocation_test_failed& e)
        {
            gtest_info << e.what() << "\nContent of error log :\n" << exception_logger.get_log();
            test_failed = true;
        }
        catch(...)
        {
            gtest_info << "unidentified exception caught during test.\nContent of error log :\n"
                       << exception_logger.get_log();
            test_failed = true;
        }

        if(hipfftw_alloc && !hipfftw_.free.may_be_used())
            throw std::runtime_error("An allocation was created but it can't be freed");
        // note: free should be stable even with nullptr
        if(hipfftw_.free.may_be_used())
            hipfftw_.free(hipfftw_alloc);
        // reenable disabled allocation kinds (ignoring returned values below)
        const std::string no_limit_str = std::to_string(std::numeric_limits<size_t>::max());
        if(pinned_host_alloc_was_disabled)
        {
#ifdef WIN32
            (void)SetEnvironmentVariable("HIPFFTW_ALLOC_LIMIT_PINNED_HOST", no_limit_str.c_str());
#else
            (void)setenv("HIPFFTW_ALLOC_LIMIT_PINNED_HOST", no_limit_str.c_str(), 1);
#endif
        }
        if(pageable_host_alloc_was_disabled)
        {
#ifdef WIN32
            (void)SetEnvironmentVariable("HIPFFTW_ALLOC_LIMIT_PAGEABLE_HOST", no_limit_str.c_str());
#else
            (void)setenv("HIPFFTW_ALLOC_LIMIT_PAGEABLE_HOST", no_limit_str.c_str(), 1);
#endif
        }

        if(test_skipped)
            GTEST_SKIP() << gtest_info.str();
        else if(test_failed)
            GTEST_FAIL() << gtest_info.str();
        if(pinned_host_alloc_was_disabled && params.alloc_arg > 0)
        {
            const std::string log_content     = exception_logger.get_log();
            const std::string expected_in_log = "Redirecting execution flow";
            if(log_content.find(expected_in_log) == std::string::npos)
            {
                GTEST_FAIL() << "no instance of \"" << expected_in_log
                             << "\" in log despite disabling pinned host allocation via "
                                "environment variable. Content of log :\n"
                             << log_content;
            }
        }
    }

    // this routine chooses values of hipfftw_malloc_params to be tested by the allocation suite
    std::vector<hipfftw_malloc_params> params_to_test_allocation()
    {
        std::vector<hipfftw_malloc_params> ret;
        // testing argument value 0 and a randomly chosen one (max 64MiB)
        constexpr size_t                      max_test_alloc_size = 1ULL << 26;
        std::ranlux24_base                    gen(random_seed);
        std::uniform_int_distribution<size_t> arg_rng(1, max_test_alloc_size);
        auto get_nonzero_arg = [&](const hipfftw_malloc_params::alloc_api_type& alloc_type,
                                   const fft_precision&                         prec) {
            if(alloc_type != hipfftw_malloc_params::alloc_api_type::none
               && alloc_type != hipfftw_malloc_params::alloc_api_type::real
               && alloc_type != hipfftw_malloc_params::alloc_api_type::complex)
                throw std::invalid_argument("unexpected alloc_type for get_nonzero_arg lambda.");
            if(prec != fft_precision_single && prec != fft_precision_double)
                throw std::invalid_argument("unexpected precision for get_nonzero_arg lambda.");
            size_t elem_size = sizeof(char);
            if(alloc_type == hipfftw_malloc_params::alloc_api_type::real)
            {
                elem_size = prec == fft_precision_single
                                ? sizeof(hipfftw_real_t<fft_precision_single>)
                                : sizeof(hipfftw_real_t<fft_precision_double>);
            }
            else
            {
                elem_size = prec == fft_precision_single
                                ? sizeof(hipfftw_complex_t<fft_precision_single>)
                                : sizeof(hipfftw_complex_t<fft_precision_double>);
            }
            size_t ret = arg_rng(gen);
            while(ret * elem_size > max_test_alloc_size || ret == 0)
                ret = arg_rng(gen);
            return ret;
        };

        for(auto api : {hipfftw_malloc_params::alloc_api_type::none,
                        hipfftw_malloc_params::alloc_api_type::real,
                        hipfftw_malloc_params::alloc_api_type::complex})
        {
            for(auto prec : {fft_precision_single, fft_precision_double})
            {
                for(auto avoid_kind : {hipfftw_malloc_params::none,
                                       hipfftw_malloc_params::pinned_host,
                                       hipfftw_malloc_params::pageable_host,
                                       static_cast<hipfftw_malloc_params::alloc_kind_disabler>(
                                           hipfftw_malloc_params::pinned_host
                                           | hipfftw_malloc_params::pageable_host)})
                {
                    for(auto arg : {size_t(0), get_nonzero_arg(api, prec)})
                    {
                        ret.push_back({arg, api, prec, avoid_kind});
                    }
                }
            }
        }
        return ret;
    }

    template <fft_precision prec>
    void test_existence_of_utility_functions()
    {
        try
        {
            // call utility functions - they need to exist but don't need to work
            const auto& hipfftw_ = hipfftw_funcs<prec>::get_instance();
            hipfftw_.print_plan(nullptr);
            hipfftw_.set_timelimit(0.0);
            hipfftw_.cost(nullptr);
            hipfftw_.flops(nullptr, nullptr, nullptr, nullptr);
            hipfftw_.cleanup();
        }
        catch(const hipfftw_undefined_function_ptr& e)
        {
            GTEST_SKIP() << "Undefined function pointers detected. Error info: " << e.what();
        }
        catch(...)
        {
            GTEST_FAIL() << "Unexpected failure";
        }
    }

    // test suite for testing hipfftw functional correctness (happy paths)
    class functional : public ::testing::TestWithParam<hipfftw_params>
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}

    public:
        static std::string TestName(const testing::TestParamInfo<functional::ParamType>& info)
        {
            return info.param.token();
        }
    };

    enum class io_label
    {
        IN,
        OUT,
    };

    enum class hipfftw_plan_creation_failure_t
    {
        invalid_args,
        unsupported_args
    };
    template <hipfftw_plan_creation_failure_t>
    std::string_view expected_log_instance;

    template <>
    std::string_view expected_log_instance<
        hipfftw_plan_creation_failure_t::invalid_args> = R"(Invalid argument)";
    template <>
    std::string_view expected_log_instance<
        hipfftw_plan_creation_failure_t::unsupported_args> = R"(Feature is not supported)";

    struct hipfftw_creation_params
    {
    private:
        template <typename T>
        struct dyn_array_bundle
        {
            T*     data = nullptr;
            size_t sz   = 0;
            void   resize(size_t new_size, bool copy_data = true)
            {
                if(new_size == sz)
                    return;
                if(new_size == 0)
                {
                    clear();
                    return;
                }
                T* temp = new T[new_size];
                if(copy_data)
                    for(auto i = 0; i < std::min(new_size, sz); i++)
                        temp[i] = data[i];
                clear();
                data = temp;
                sz   = new_size;
            }
            void clear()
            {
                delete[] data;
                data = nullptr;
                sz   = 0;
            }
            ~dyn_array_bundle()
            {
                clear();
            }

            dyn_array_bundle() = default;
            // move constructor
            dyn_array_bundle(dyn_array_bundle&& other)
            {
                data       = other.data;
                sz         = other.sz;
                other.data = nullptr;
                other.sz   = 0;
            }
            // move assignment
            dyn_array_bundle& operator=(dyn_array_bundle&& other)
            {
                std::swap(data, other.data);
                std::swap(sz, other.sz);
            }
            // (deep) copy constructor
            dyn_array_bundle(const dyn_array_bundle& other)
            {
                resize(other.sz);
                for(auto i = 0; i < sz; i++)
                    data[i] = other.data[i];
            }
            // (deep) copy assignment
            dyn_array_bundle& operator=(const dyn_array_bundle& rhs)
            {
                resize(rhs.sz);
                for(auto i = 0; i < sz; i++)
                    data[i] = rhs.data[i];
                return *this;
            }
        };

        int                         rank = 0;
        dyn_array_bundle<int>       int_lengths;
        dyn_array_bundle<ptrdiff_t> ptrdiff_lengths;
        dyn_array_bundle<char>      input;
        dyn_array_bundle<char>      output;
        int                         sign  = 0;
        unsigned                    flags = std::numeric_limits<unsigned>::max();
        // not an arg to plan creation per se, but used in tests and checks to
        // know whether to account for 'output' at all or not
        fft_result_placement placement = fft_placement_inplace;

        template <typename T>
        const dyn_array_bundle<T>& get_lengths() const
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            if constexpr(std::is_same_v<T, int>)
                return int_lengths;
            else
                return ptrdiff_lengths;
        }
        template <typename T>
        dyn_array_bundle<T>& get_lengths()
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            if constexpr(std::is_same_v<T, int>)
                return int_lengths;
            else
                return ptrdiff_lengths;
        }
        template <typename T>
        void set_to_array_if_representable_as(const std::vector<ptrdiff_t>& new_lengths)
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            auto& arr_bundle = get_lengths<T>();
            if constexpr(!std::is_same_v<T, ptrdiff_t>)
            {
                if(std::any_of(new_lengths.begin(), new_lengths.end(), [](const ptrdiff_t& val) {
                       return val > std::numeric_limits<T>::max()
                              || val < std::numeric_limits<T>::lowest();
                   }))
                {
                    arr_bundle.clear();
                    return;
                }
            }
            arr_bundle.resize(new_lengths.size(), false);
            for(auto i = 0; i < new_lengths.size(); i++)
                arr_bundle.data[i] = static_cast<T>(new_lengths[i]);
        }

        bool is_valid_as_io(const dyn_array_bundle<char>& io) const
        {
            // valid if not nullptr unless using FFTW_ESTIMATE or FFTW_WISDOM_ONLY
            return io.data || (flags & FFTW_ESTIMATE || flags & FFTW_WISDOM_ONLY);
        }
        // lengths are always representale as ptrdiff_t
        void append_length(std::ostringstream& oss) const
        {
            oss << "_lengths";
            const auto& arr = get_lengths<ptrdiff_t>();
            if(!arr.data)
                oss << "_none";
            else
            {
                for(auto i = 0; i < arr.sz; i++)
                {
                    oss << (arr.data[i] < 0 ? "_negative_" : "_") << std::abs(arr.data[i]);
                }
            }
        };

        static std::ranlux24_base& get_pseudo_rng()
        {
            static std::ranlux24_base gen(random_seed);
            return gen;
        }
        // static private functions common to public checkers and static randomizers
        static bool rank_is_valid(int r)
        {
            return r > 0;
        }
        template <typename T>
        static bool lengths_are_valid(const T* len, size_t len_size)
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            bool valid_len  = len_size > 0 && len;
            T    min_stride = 1; // checking that default strides/dist do not overflow for type T
            for(int i = len_size - 1; valid_len && i >= 0; i--)
            {
                if(len[i] <= 0)
                    valid_len = false;
                else
                {
                    const T min_multiplier = i == len_size - 1 ? len[i] / 2 + 1 : len[i];
                    if(min_stride > std::numeric_limits<T>::max() / min_multiplier)
                    {
                        valid_len = false;
                    }
                    min_stride *= min_multiplier;
                }
            }
            return valid_len;
        }
        static bool sign_is_valid(int s)
        {
            return s == FFTW_FORWARD || s == FFTW_BACKWARD;
        }
        constexpr static unsigned valid_flags_mask
            = FFTW_WISDOM_ONLY | FFTW_MEASURE | FFTW_DESTROY_INPUT | FFTW_UNALIGNED
              | FFTW_CONSERVE_MEMORY | FFTW_EXHAUSTIVE | FFTW_PRESERVE_INPUT | FFTW_PATIENT
              | FFTW_ESTIMATE;
        static bool flags_are_valid(unsigned f)
        {
            return (f & valid_flags_mask) == f;
        }

    public:
        hipfftw_creation_params()  = default;
        ~hipfftw_creation_params() = default;
        // default moves and copies are fine
        hipfftw_creation_params(hipfftw_creation_params&& other) = default;
        hipfftw_creation_params& operator=(hipfftw_creation_params&& other) = default;
        hipfftw_creation_params(const hipfftw_creation_params& other)       = default;
        hipfftw_creation_params& operator=(const hipfftw_creation_params& rhs) = default;

        void set_lengths(const std::vector<ptrdiff_t>& new_lengths)
        {
            set_to_array_if_representable_as<int>(new_lengths);
            set_to_array_if_representable_as<ptrdiff_t>(new_lengths);
        }
        void set_rank(int new_rank)
        {
            rank = new_rank;
        }
        template <io_label io>
        void define_data()
        {
            // make sure {input,output}.data != nullptr
            if constexpr(io == io_label::IN)
                input.resize(sizeof(hipfftw_complex_t<fft_precision_double>), false);
            else
                output.resize(sizeof(hipfftw_complex_t<fft_precision_double>), false);
        }
        template <io_label io>
        void clear_data()
        {
            // sets {input,output}.data to  nullptr
            if constexpr(io == io_label::IN)
                input.clear();
            else
                output.clear();
        }
        void set_placement(fft_result_placement new_placement)
        {
            placement = new_placement;
        }
        void set_sign(int new_sign)
        {
            sign = new_sign;
        }
        void set_flags(unsigned new_flags)
        {
            flags = new_flags;
        }
        void clear_io()
        {
            clear_data<io_label::IN>();
            clear_data<io_label::OUT>();
        }
        void clear_lengths()
        {
            get_lengths<int>().clear();
            get_lengths<ptrdiff_t>().clear();
        }
        // getters
        int get_rank() const
        {
            return rank;
        }
        int get_sign() const
        {
            return sign;
        }
        unsigned get_flags() const
        {
            return flags;
        }
        template <typename T>
        T* get_length_array() const
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            return get_lengths<T>().data;
        }
        template <io_label io>
        void* get_ptr() const
        {
            if(io == io_label::IN || placement == fft_placement_inplace)
                return input.data;
            else
                return output.data;
        }
        template <typename T>
        size_t get_length_sz() const
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            return get_lengths<T>().sz;
        }
        // validity checkers
        bool has_valid_rank() const
        {
            return rank_is_valid(rank);
        }
        template <typename T>
        bool has_valid_lengths() const
        {
            static_assert(std::is_same_v<T, int> || std::is_same_v<T, ptrdiff_t>);
            if(!rank_is_valid(rank))
                return false; // impossible to validate lengths for an invalid rank
            const auto& array = get_lengths<T>();
            return array.sz == rank && lengths_are_valid(array.data, array.sz);
        }
        template <io_label io>
        bool has_valid_data() const
        {
            if(io == io_label::IN || placement == fft_placement_inplace)
                return is_valid_as_io(input);
            else
                return is_valid_as_io(output);
        }
        bool has_valid_sign() const
        {
            return sign_is_valid(sign);
        }
        bool has_valid_flags() const
        {
            return flags_are_valid(flags);
        }
        template <bool sign_matters>
        bool is_valid_for_basic_plan_creation() const
        {
            return has_valid_rank() && has_valid_lengths<int>() && has_valid_data<io_label::IN>()
                   && has_valid_data<io_label::OUT>() && (!sign_matters || has_valid_sign())
                   && has_valid_flags();
        }
        template <bool for_real_fwd, bool for_real_bwd>
        bool is_supported_for_basic_plan_creation() const
        {
            constexpr bool sign_does_matter = !for_real_fwd && !for_real_bwd;
            if(!is_valid_for_basic_plan_creation<sign_does_matter>())
                return false;
            if(rank > 3)
                return false;
            if(flags & FFTW_WISDOM_ONLY)
                return false;
            if constexpr(for_real_bwd)
            {
                if(rank > 1 && (flags & FFTW_PRESERVE_INPUT))
                    return false;
            }
            return true;
        }

        bool is_invalid_for_some_plan_creation() const
        {
            return !is_valid_for_basic_plan_creation<true>()
                   || !is_valid_for_basic_plan_creation<false>();
        }
        bool is_unsupported_for_some_plan_creation() const
        {
            // c2c apis are common to either direction, real apis are not
            constexpr bool for_real_fwd = true;
            constexpr bool for_real_bwd = true;
            return !is_supported_for_basic_plan_creation<!for_real_fwd, !for_real_bwd>()
                   || !is_supported_for_basic_plan_creation<for_real_fwd, !for_real_bwd>()
                   || !is_supported_for_basic_plan_creation<!for_real_fwd, for_real_bwd>();
        }
        // converting to string
        std::string token() const
        {
            std::ostringstream ret;
            ret << "rank" << (rank < 0 ? "_negative_" : "_") << std::abs(rank);
            append_length(ret);

            ret << "_in_ptr" << (input.data ? "_not_" : "_") << "nullptr";
            ret << "_out_ptr";
            if(placement == fft_placement_inplace)
            {
                ret << "_same_as_in_ptr";
            }
            else
            {
                ret << (output.data ? "_not_" : "_") << "nullptr";
            }
            ret << "_sign" << (sign < 0 ? "_negative_" : "_") << std::abs(sign);
            ret << "_flags_" << flags;
            return ret.str();
        }
        // static randomizers
        // Note: albeit not supported, ranks > 3 are "valid" rank argument
        // --> limiting rank value to max of 10 by default to avoid ridiculously long
        // lengths possibly created in automated parameter generations;
        template <bool validity_flag,
                  int  min_rank          = validity_flag ? 1 : std::numeric_limits<int>::lowest(),
                  int  max_rank          = validity_flag ? 10 : 0,
                  std::enable_if_t<(min_rank <= max_rank) && (!validity_flag || min_rank > 0)
                                       && (validity_flag || max_rank <= 0),
                                   bool> = true>
        static int get_random_rank()
        {
            std::uniform_int_distribution<int> rank_rng(min_rank, max_rank);

            auto ret = rank_rng(get_pseudo_rng());
            if(rank_is_valid(ret) != validity_flag)
            {
                throw std::runtime_error(
                    "failed to generate a rank value of desired validity randomly");
            }
            return ret;
        }
        template <bool validity_flag>
        static std::vector<ptrdiff_t> get_random_lengths_for_rank(int       r,
                                                                  ptrdiff_t max_abs_len
                                                                  = std::numeric_limits<int>::max(),
                                                                  ptrdiff_t min_abs_len = 0)
        {
            std::vector<ptrdiff_t> ret;
            // cannot generate lengths for invalid ranks --> return empty lengths in that case
            if(!rank_is_valid(r))
                return ret;
            if(min_abs_len < 0 || max_abs_len < 0 || min_abs_len > max_abs_len)
                throw std::invalid_argument("invalid bounds used for get_random_lengths_for_rank");
            // generate values that are all representable as integers
            auto&                                    pseudo_rng = get_pseudo_rng();
            std::uniform_int_distribution<ptrdiff_t> length_rng(min_abs_len, max_abs_len);
            // setter lambda
            auto set_random_len = [&]() {
                for(auto& l : ret)
                {
                    const ptrdiff_t val = length_rng(pseudo_rng);
                    if constexpr(validity_flag)
                        l = val;
                    else
                    {
                        if(pseudo_rng() % 2)
                            l = -val;
                        else
                            l = val;
                    }
                }
            };

            ret.resize(r);
            set_random_len();
            while(lengths_are_valid(ret.data(), r) != validity_flag)
                set_random_len();

            return ret;
        }
        template <bool validity_flag>
        static int get_random_sign()
        {
            std::uniform_int_distribution<int> sign_rng(std::numeric_limits<int>::lowest(),
                                                        std::numeric_limits<int>::max());

            auto tmp = sign_rng(get_pseudo_rng());
            if(validity_flag)
                return tmp % 2 == 0 ? FFTW_FORWARD : FFTW_BACKWARD;

            while(sign_is_valid(tmp))
                tmp = sign_rng(get_pseudo_rng());
            return tmp;
        }
        template <bool validity_flag>
        static int get_random_flags()
        {
            std::uniform_int_distribution<unsigned> flags_rng(
                std::numeric_limits<unsigned>::lowest(), std::numeric_limits<unsigned>::max());

            auto tmp = flags_rng(get_pseudo_rng());
            if(validity_flag)
            {
                tmp &= valid_flags_mask;
                if(!flags_are_valid(tmp))
                    throw std::runtime_error("failed to create random valid flags");
                return tmp;
            }
            while(flags_are_valid(tmp))
                tmp = flags_rng(get_pseudo_rng());
            return tmp;
        }

        template <hipfftw_plan_creation_failure_t failure_reason,
                  bool                            is_real_fwd,
                  bool                            is_real_bwd>
        bool expect_failure_for_basic_plan_creation() const
        {
            static_assert(!is_real_fwd || !is_real_bwd);
            static_assert(failure_reason == hipfftw_plan_creation_failure_t::invalid_args
                          || failure_reason == hipfftw_plan_creation_failure_t::unsupported_args);
            if constexpr(failure_reason == hipfftw_plan_creation_failure_t::invalid_args)
                return !is_valid_for_basic_plan_creation<(!is_real_fwd && !is_real_bwd)>();
            else
                return !is_supported_for_basic_plan_creation<is_real_fwd, is_real_bwd>();
        }
    };

    class invalid_arg_for : public ::testing::TestWithParam<hipfftw_creation_params>
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}

    public:
        static std::string TestName(const testing::TestParamInfo<invalid_arg_for::ParamType>& info)
        {
            return info.param.token();
        }
    };

    class unsupported_arg_for : public ::testing::TestWithParam<hipfftw_creation_params>
    {
    protected:
        void SetUp() override {}
        void TearDown() override {}

    public:
        static std::string
            TestName(const testing::TestParamInfo<unsupported_arg_for::ParamType>& info)
        {
            return info.param.token();
        }
    };

    std::vector<hipfftw_params> params_for_accuracy_tests(size_t max_double_data_byte_size,
                                                          size_t num_lengths)
    {
        // limiting to lengths <= 1024 per dimension
        std::ranlux24_base                    gen(random_seed);
        std::uniform_int_distribution<size_t> size_rng(1, 1024);
        // lexicographically-sorted set to return
        std::vector<std::vector<size_t>> set_of_test_lengths;
        while(set_of_test_lengths.size() < num_lengths)
        {
            // alternate between 1D, 2D, and 3D sizes
            const size_t        dim = 1 + (set_of_test_lengths.size() % 3);
            std::vector<size_t> to_add(dim, 0);
            for(auto& length : to_add)
                length = size_rng(gen);
            if(2 * product(to_add.begin(), to_add.end()) * sizeof(double)
               > max_double_data_byte_size)
                continue;
            auto it
                = std::lower_bound(set_of_test_lengths.begin(), set_of_test_lengths.end(), to_add);
            if(it == set_of_test_lengths.end() || *it != to_add)
            {
                set_of_test_lengths.insert(it, to_add);
            }
        }
        std::vector<hipfftw_params> ret;
        // set_of_lengths assumed not to contain duplicates
        for(const auto& test_lengths : set_of_test_lengths)
        {
            for(auto type : {fft_transform_type_complex_forward, fft_transform_type_real_forward})
            {
                for(auto prec : {fft_precision_single, fft_precision_double})
                {
                    for(auto placement : {fft_placement_inplace, fft_placement_notinplace})
                    {
                        hipfftw_params to_add;
                        to_add.length         = test_lengths;
                        to_add.transform_type = type;
                        to_add.precision      = prec;
                        to_add.placement      = placement;
                        // all other parameter members are left as default
                        to_add.validate();

                        const double roll = hash_prob(random_seed, to_add.token());
                        const double run_prob
                            = test_prob * (to_add.is_real() ? real_prob_factor : 1.0);

                        if(roll > run_prob)
                        {
                            if(verbose > 4)
                            {
                                std::cout << "Test skipped: (roll=" << roll << " > " << run_prob
                                          << ")\n";
                            }
                            continue;
                        }
                        ret.emplace_back(to_add);
                    }
                }
            }
        }
        return ret;
    }

    ptrdiff_t valid_int_length_threshold(const int& rank)
    {
        if(rank < 1)
            throw std::invalid_argument(
                "invalid rank used for calculating valid_int_length_threshold");
        // (rough) approximate threshold for absolute values of lengths
        // to prevent overflows in strides/distances
        return rank == 1 ? static_cast<ptrdiff_t>(std::numeric_limits<int>::max())
                         : static_cast<ptrdiff_t>(
                             std::floor(std::pow(std::numeric_limits<int>::max(), 1.0 / rank)));
    }

    std::vector<hipfftw_creation_params> invalid_params_for_plan_creation()
    {
        typedef std::pair<bool, bool> bool_pair;
        constexpr bool                valid_value = true; // constexpr used for readability

        const auto invalid_rank = hipfftw_creation_params::get_random_rank<!valid_value>();
        std::vector<hipfftw_creation_params> ret;
        hipfftw_creation_params              to_add;
        std::vector<std::string>             tokens;
        for(int rank : {1, 2, 3, invalid_rank})
        {
            std::vector<std::vector<ptrdiff_t>> range_of_lengths;
            if(rank > 0)
            {
                const ptrdiff_t len_threshold = valid_int_length_threshold(rank);
                // valid integer lengths with strides/distances that do not overflow
                range_of_lengths.emplace_back(
                    hipfftw_creation_params::get_random_lengths_for_rank<valid_value>(
                        rank, len_threshold));
                if(rank > 1)
                {
                    // valid integer lengths with strides/distances that likely overflow
                    range_of_lengths.emplace_back(
                        hipfftw_creation_params::get_random_lengths_for_rank<valid_value>(
                            rank, std::numeric_limits<int>::max(), len_threshold));
                }
                // invalid integer lengths (nonzero)
                range_of_lengths.emplace_back(
                    hipfftw_creation_params::get_random_lengths_for_rank<!valid_value>(rank));
                // invalid integer lengths (zeros)
                range_of_lengths.emplace_back(
                    hipfftw_creation_params::get_random_lengths_for_rank<!valid_value>(rank, 0, 0));
            }
            // empty lengths
            range_of_lengths.emplace_back(std::vector<ptrdiff_t>());
            for(const auto& lengths : range_of_lengths)
            {
                for(auto placement : {fft_placement_inplace, fft_placement_notinplace})
                {
                    std::vector<bool_pair> set_io_range
                        = {bool_pair(false, false), bool_pair(true, false)};
                    if(placement != fft_placement_notinplace)
                    {
                        set_io_range.push_back({false, true});
                        set_io_range.push_back({true, true});
                    }
                    for(auto set_io : set_io_range)
                    {
                        const auto valid_sign
                            = hipfftw_creation_params::get_random_sign<valid_value>();
                        const auto invalid_sign
                            = hipfftw_creation_params::get_random_sign<!valid_value>();
                        for(auto sign : {valid_sign, invalid_sign})
                        {
                            const auto valid_flags
                                = hipfftw_creation_params::get_random_flags<valid_value>();
                            const auto invalid_flags
                                = hipfftw_creation_params::get_random_flags<!valid_value>();
                            for(auto flags : {valid_flags, invalid_flags})
                            {
                                to_add.set_rank(rank);
                                to_add.set_lengths(lengths);
                                to_add.set_placement(placement);
                                if(set_io.first)
                                    to_add.define_data<io_label::IN>();
                                else
                                    to_add.clear_data<io_label::IN>();
                                if(set_io.second)
                                    to_add.define_data<io_label::OUT>();
                                else
                                    to_add.clear_data<io_label::OUT>();
                                to_add.set_sign(sign);
                                to_add.set_flags(flags);
                                // save only combos that are invalid for some plan
                                // creation and that were not previously generated
                                if(to_add.is_invalid_for_some_plan_creation())
                                {
                                    const std::string test_token = to_add.token();
                                    auto              it         = std::lower_bound(
                                        tokens.begin(), tokens.end(), test_token);
                                    if(it != tokens.end() && *it == test_token)
                                        continue; // already generated
                                    tokens.insert(it, test_token);
                                    ret.emplace_back(to_add);
                                }
                            }
                        }
                    }
                }
            }
        }
        return ret;
    }

    std::vector<hipfftw_creation_params> unsupported_params_for_plan_creation()
    {
        std::vector<hipfftw_creation_params> ret;
        hipfftw_creation_params              to_add;
        // always define input and output, as we want to avoid "invalid arguments"
        // for what's being tested here.
        to_add.define_data<io_label::IN>();
        to_add.define_data<io_label::OUT>();
        constexpr bool valid_value = true; // constexpr used for readability
        // FFTW_WISDOM_ONLY flag is not supported
        for(auto rank : {1, 2, 3})
        {
            for(auto placement : {fft_placement_inplace, fft_placement_notinplace})
            {
                for(auto sign : {FFTW_FORWARD, FFTW_BACKWARD})
                {
                    to_add.set_rank(rank);
                    to_add.set_lengths(
                        hipfftw_creation_params::get_random_lengths_for_rank<valid_value>(
                            rank, valid_int_length_threshold(rank)));
                    to_add.set_placement(placement);
                    to_add.set_sign(sign);
                    to_add.set_flags(FFTW_WISDOM_ONLY
                                     | hipfftw_creation_params::get_random_flags<valid_value>());
                    if(!to_add.is_invalid_for_some_plan_creation())
                        ret.emplace_back(to_add);
                }
            }
        }
        // FFTW_PRESERVE_INPUT is not supported for out-of-place 2D/3D c2r
        for(auto rank : {2, 3})
        {
            to_add.set_rank(rank);
            to_add.set_lengths(hipfftw_creation_params::get_random_lengths_for_rank<valid_value>(
                rank, valid_int_length_threshold(rank)));
            to_add.set_placement(fft_placement_notinplace);
            to_add.set_sign(FFTW_BACKWARD);
            to_add.set_flags(FFTW_PRESERVE_INPUT
                             | hipfftw_creation_params::get_random_flags<valid_value>());
            if(!to_add.is_invalid_for_some_plan_creation())
                ret.emplace_back(to_add);
        }
        // ranks > 3 are not supported
        constexpr int min_unsupported_rank = 4;
        for(auto placement : {fft_placement_inplace, fft_placement_notinplace})
        {
            for(auto sign : {FFTW_FORWARD, FFTW_BACKWARD})
            {
                const int unsupported_rank
                    = hipfftw_creation_params::get_random_rank<valid_value, min_unsupported_rank>();
                to_add.set_rank(unsupported_rank);
                to_add.set_lengths(
                    hipfftw_creation_params::get_random_lengths_for_rank<valid_value>(
                        unsupported_rank, valid_int_length_threshold(unsupported_rank)));
                to_add.set_placement(placement);
                to_add.set_sign(sign);
                to_add.set_flags(hipfftw_creation_params::get_random_flags<valid_value>());
                if(!to_add.is_invalid_for_some_plan_creation())
                    ret.emplace_back(to_add);
            }
        }
        return ret;
    }

    template <fft_precision prec, hipfftw_plan_creation_failure_t failure_reason>
    void expect_plan_creation_failure(const hipfftw_creation_params& params)
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        static_assert(failure_reason == hipfftw_plan_creation_failure_t::invalid_args
                      || failure_reason == hipfftw_plan_creation_failure_t::unsupported_args);
        const auto& hipfftw_ = hipfftw_funcs<prec>::get_instance();
        // plan attempted for creation
        typename hipfftw_trait<prec>::plan_t hipfftw_plan = nullptr;
        // exception-logging-related variables
        hipfftw_exception_logger exception_logger;
        size_t                   test_counter = 0;

        auto throw_test_failure_if_not_nullptr = [&](const std::string& which_plan_creation) {
            test_counter++;
            if(hipfftw_plan)
                throw std::runtime_error(which_plan_creation + " actually created a plan");
        };
        std::ostringstream gtest_info;
        bool               test_skipped = false;
        bool               test_failed  = false;

        constexpr bool is_real_bwd   = true; // constexpr used for readability
        constexpr bool is_real_fwd   = true; // constexpr used for readability
        const int      rank          = params.get_rank();
        const int*     int_len_array = params.get_length_array<int>();
        const size_t   int_len_sz    = params.get_length_sz<int>();
        void*          in            = params.get_ptr<io_label::IN>();
        void*          out           = params.get_ptr<io_label::OUT>();
        const int      sign          = params.get_sign();
        const unsigned flags         = params.get_flags();
        try
        {
            if(0 < rank && rank <= 3 && int_len_array && int_len_sz == rank)
            {
                switch(rank)
                {
                case 1:
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     !is_real_fwd,
                                                                     !is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_1d(int_len_array[0],
                                                   static_cast<hipfftw_complex_t<prec>*>(in),
                                                   static_cast<hipfftw_complex_t<prec>*>(out),
                                                   sign,
                                                   flags);
                        throw_test_failure_if_not_nullptr("plan_dft_1d");
                    }
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     is_real_fwd,
                                                                     !is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_r2c_1d(int_len_array[0],
                                                       static_cast<hipfftw_real_t<prec>*>(in),
                                                       static_cast<hipfftw_complex_t<prec>*>(out),
                                                       flags);
                        throw_test_failure_if_not_nullptr("plan_dft_r2c_1d");
                    }
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     !is_real_fwd,
                                                                     is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_c2r_1d(int_len_array[0],
                                                       static_cast<hipfftw_complex_t<prec>*>(in),
                                                       static_cast<hipfftw_real_t<prec>*>(out),
                                                       flags);
                        throw_test_failure_if_not_nullptr("plan_dft_c2r_1d");
                    }
                    break;
                case 2:
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     !is_real_fwd,
                                                                     !is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_2d(int_len_array[0],
                                                   int_len_array[1],
                                                   static_cast<hipfftw_complex_t<prec>*>(in),
                                                   static_cast<hipfftw_complex_t<prec>*>(out),
                                                   sign,
                                                   flags);
                        throw_test_failure_if_not_nullptr("plan_dft_2d");
                    }
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     is_real_fwd,
                                                                     !is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_r2c_2d(int_len_array[0],
                                                       int_len_array[1],
                                                       static_cast<hipfftw_real_t<prec>*>(in),
                                                       static_cast<hipfftw_complex_t<prec>*>(out),
                                                       flags);
                        throw_test_failure_if_not_nullptr("plan_dft_r2c_2d");
                    }
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     !is_real_fwd,
                                                                     is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_c2r_2d(int_len_array[0],
                                                       int_len_array[1],
                                                       static_cast<hipfftw_complex_t<prec>*>(in),
                                                       static_cast<hipfftw_real_t<prec>*>(out),
                                                       flags);
                        throw_test_failure_if_not_nullptr("plan_dft_c2r_2d");
                    }

                    break;
                case 3:
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     !is_real_fwd,
                                                                     !is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_3d(int_len_array[0],
                                                   int_len_array[1],
                                                   int_len_array[2],
                                                   static_cast<hipfftw_complex_t<prec>*>(in),
                                                   static_cast<hipfftw_complex_t<prec>*>(out),
                                                   sign,
                                                   flags);
                        throw_test_failure_if_not_nullptr("plan_dft_3d");
                    }
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     is_real_fwd,
                                                                     !is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_r2c_3d(int_len_array[0],
                                                       int_len_array[1],
                                                       int_len_array[2],
                                                       static_cast<hipfftw_real_t<prec>*>(in),
                                                       static_cast<hipfftw_complex_t<prec>*>(out),
                                                       flags);
                        throw_test_failure_if_not_nullptr("plan_dft_r2c_3d");
                    }
                    if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                                     !is_real_fwd,
                                                                     is_real_bwd>())
                    {
                        hipfftw_plan
                            = hipfftw_.plan_dft_c2r_3d(int_len_array[0],
                                                       int_len_array[1],
                                                       int_len_array[2],
                                                       static_cast<hipfftw_complex_t<prec>*>(in),
                                                       static_cast<hipfftw_real_t<prec>*>(out),
                                                       flags);
                        throw_test_failure_if_not_nullptr("plan_dft_c2r_3d");
                    }
                    break;
                default:
                    std::runtime_error("unexpected supposedly-valid rank encountered");
                }
            }
            if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                             !is_real_fwd,
                                                             !is_real_bwd>())
            {
                hipfftw_plan = hipfftw_.plan_dft(rank,
                                                 int_len_array,
                                                 static_cast<hipfftw_complex_t<prec>*>(in),
                                                 static_cast<hipfftw_complex_t<prec>*>(out),
                                                 sign,
                                                 flags);
                throw_test_failure_if_not_nullptr("plan_dft");
            }
            if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                             is_real_fwd,
                                                             !is_real_bwd>())
            {
                hipfftw_plan = hipfftw_.plan_dft_r2c(rank,
                                                     int_len_array,
                                                     static_cast<hipfftw_real_t<prec>*>(in),
                                                     static_cast<hipfftw_complex_t<prec>*>(out),
                                                     flags);
                throw_test_failure_if_not_nullptr("plan_dft_r2c");
            }
            if(params.expect_failure_for_basic_plan_creation<failure_reason,
                                                             !is_real_fwd,
                                                             is_real_bwd>())
            {
                hipfftw_plan = hipfftw_.plan_dft_c2r(rank,
                                                     int_len_array,
                                                     static_cast<hipfftw_complex_t<prec>*>(in),
                                                     static_cast<hipfftw_real_t<prec>*>(out),
                                                     flags);
                throw_test_failure_if_not_nullptr("plan_dft_c2r");
            }
        }
        catch(const hipfftw_undefined_function_ptr& e)
        {
            gtest_info << "undefined function pointers detected. Error info: " << e.what();
            test_skipped = true;
        }
        catch(const std::runtime_error e)
        {
            gtest_info << e.what() << "\n content of error log :\n" << exception_logger.get_log();
            test_failed = true;
        }
        catch(...)
        {
            gtest_info << "unidentified exception caught during test.\n content of error log :\n"
                       << exception_logger.get_log();
            test_failed = true;
        }
        if(hipfftw_plan && !hipfftw_.destroy_plan.may_be_used())
            throw std::runtime_error("A plan was created but it can't be destroyed");
        // note: destruction should be stable even with nullptr
        if(hipfftw_.destroy_plan.may_be_used())
            hipfftw_.destroy_plan(hipfftw_plan);
        if(test_skipped)
            GTEST_SKIP() << gtest_info.str();
        else if(test_failed)
            GTEST_FAIL() << gtest_info.str();
        // count number of log instances
        size_t                 num_log_instance = 0;
        const std::string      log_content      = exception_logger.get_log();
        std::string::size_type pos              = 0;
        while((pos = log_content.find(expected_log_instance<failure_reason>, pos))
              != std::string::npos)
        {
            num_log_instance++;
            pos += expected_log_instance<failure_reason>.length();
        }
        if(num_log_instance != test_counter)
            GTEST_FAIL() << "incorrect number of log instances detected (expeted " << test_counter
                         << " instance of \""
                         << expected_log_instance<failure_reason> << "\"). Content of error log :\n"
                         << log_content;
    }

} // hipfftw test details' anonymous namespace

TEST(hipfftw_test, utility_functions)
{
    test_existence_of_utility_functions<fft_precision_single>();
    test_existence_of_utility_functions<fft_precision_double>();
}

TEST_P(allocation, malloc_write_read_free)
{
    const auto params = GetParam();
    if(params.prec == fft_precision_single)
        malloc_write_read_free_test<fft_precision_single>(params);
    else
        malloc_write_read_free_test<fft_precision_double>(params);
}

INSTANTIATE_TEST_SUITE_P(hipfftw_test,
                         allocation,
                         ::testing::ValuesIn(params_to_test_allocation()),
                         allocation::TestName);

// Test for comparison between FFTW and hipfftw.
TEST_P(functional, accuracy_vs_fftw)
{
    hipfftw_params params(GetParam());

    params.validate();

    if(!params.valid(verbose))
    {
        GTEST_SKIP() << "Invalid parameters, skipping this test. Test token: " << params.token();
    }

    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    if(test_name.find(params.token()) == std::string::npos)
        GTEST_FAIL() << "inconsistency of test parameter" << std::endl;

    try
    {
        const bool do_round_trip = params.is_forward();
        switch(params.precision)
        {
        case fft_precision_single:
            fft_vs_reference_impl<float, hipfftw_params>(params, do_round_trip);
            break;
        case fft_precision_double:
            fft_vs_reference_impl<double, hipfftw_params>(params, do_round_trip);
            break;
        case fft_precision_half:
            throw fft_params::unimplemented_exception("Half precision is not supported by hipfftw");
            break;
        default:
            throw std::runtime_error("Invalid precision");
            break;
        }
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        // explicitly clear cache
        last_cpu_fft_data = last_cpu_fft_cache();
        GTEST_SKIP() << e.msg;
    }
    catch(const hipfftw_undefined_function_ptr& e)
    {
        GTEST_SKIP() << "Undefined function pointers detected. Error info: " << e.what();
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(const fft_params::unimplemented_exception& e)
    {
        GTEST_SKIP() << "Unimplemented exception: " << e.what();
    }
    catch(const hipfftw_params::logic_error& e)
    {
        GTEST_FAIL() << "hipfftw logic error caught: " << e.what();
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
    catch(std::runtime_error& e)
    {
        GTEST_FAIL() << "Runtime error caught: " << e.what();
    }
    catch(...)
    {
        GTEST_FAIL() << "Unknown exception caught during test of token " << params.token();
    }

    SUCCEED();
}

// note: params_for_accuracy_tests will create 8 instances for every length(s) to be tests
// (2 precisions, 2 types of transforms, 2 types of I/O placement)
// num_parameters_for_functional_tests = 8*num_lengths_for_functional_tests
static constexpr size_t num_lengths_for_functional_tests   = 512;
static constexpr size_t max_byte_size_for_functional_tests = 512 * 1024 * 1024;

INSTANTIATE_TEST_SUITE_P(hipfftw_test,
                         functional,
                         ::testing::ValuesIn(params_for_accuracy_tests(
                             max_byte_size_for_functional_tests, num_lengths_for_functional_tests)),
                         functional::TestName);

TEST_P(invalid_arg_for, create_plan)
{
    hipfftw_creation_params params(GetParam());
    if(!params.is_invalid_for_some_plan_creation())
        GTEST_SKIP() << "Parameters are not invalid for any plan creation";

    expect_plan_creation_failure<fft_precision_single,
                                 hipfftw_plan_creation_failure_t::invalid_args>(params);
    expect_plan_creation_failure<fft_precision_double,
                                 hipfftw_plan_creation_failure_t::invalid_args>(params);
}

INSTANTIATE_TEST_SUITE_P(hipfftw_test,
                         invalid_arg_for,
                         ::testing::ValuesIn(invalid_params_for_plan_creation()),
                         invalid_arg_for::TestName);

TEST_P(unsupported_arg_for, create_plan)
{
    hipfftw_creation_params params(GetParam());
    // don't feed invalid parameters accidentally
    if(params.is_invalid_for_some_plan_creation())
        GTEST_SKIP() << "Parameters are not valid for some plan creation. Parameters are "
                     << params.token();
    if(!params.is_unsupported_for_some_plan_creation())
        GTEST_SKIP() << "Parameters are not unsupported for any plan creation";

    expect_plan_creation_failure<fft_precision_single,
                                 hipfftw_plan_creation_failure_t::unsupported_args>(params);
    expect_plan_creation_failure<fft_precision_double,
                                 hipfftw_plan_creation_failure_t::unsupported_args>(params);
}

INSTANTIATE_TEST_SUITE_P(hipfftw_test,
                         unsupported_arg_for,
                         ::testing::ValuesIn(unsupported_params_for_plan_creation()),
                         unsupported_arg_for::TestName);
