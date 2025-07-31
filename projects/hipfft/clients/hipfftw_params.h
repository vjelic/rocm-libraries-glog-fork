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

#ifndef HIPFFTW_PARAMS_H
#define HIPFFTW_PARAMS_H

#include "../shared/arithmetic.h"
#include "../shared/fft_params.h"
#include <algorithm>
#include <fftw3.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef WIN32
#include <windows.h>
// psapi.h requires windows.h to be included first
#include <psapi.h>
typedef HMODULE LIB_HANDLE_T;
#else
#include <dlfcn.h>
#include <link.h>
typedef void* LIB_HANDLE_T;
#endif

template <fft_precision prec>
struct hipfftw_trait;
template <>
struct hipfftw_trait<fft_precision_single>
{
    using plan_t    = fftwf_plan;
    using complex_t = fftwf_complex;
    using real_t    = float;
};
template <>
struct hipfftw_trait<fft_precision_double>
{
    using plan_t    = fftw_plan;
    using complex_t = fftw_complex;
    using real_t    = double;
};

template <fft_precision prec>
using hipfftw_real_t = typename hipfftw_trait<prec>::real_t;
template <fft_precision prec>
using hipfftw_complex_t = typename hipfftw_trait<prec>::complex_t;

// singleton class encapsulating the dynamically-loaded hipfftw library
class dynamically_loaded_hipfftw
{
private:
    LIB_HANDLE_T       lib_handle;
    std::ostringstream load_error_info;

    dynamically_loaded_hipfftw()
    {
        const std::string lib_basename = "hipfftw";
#ifdef WIN32
        const std::string lib_fullame = lib_basename + ".dll";
        lib_handle                    = LoadLibraryA(lib_fullame.c_str());
#else
        const std::string lib_fullame = "lib" + lib_basename + ".so";
        lib_handle                    = dlopen(lib_fullame.c_str(), RTLD_LAZY);
#endif
        load_error_info.clear();
        if(!lib_handle)
        {
            load_error_info << "failed to open library " << lib_fullame;
#ifdef WIN32
            load_error_info << ". System's error code = " << GetLastError();
#else
            load_error_info << ". System's error message = " << dlerror();
#endif
            // do not throw from here to ease exception handling
        }
    }
    /* disable copies and moves */
    dynamically_loaded_hipfftw(const dynamically_loaded_hipfftw&) = delete;
    dynamically_loaded_hipfftw(dynamically_loaded_hipfftw&&)      = delete;
    dynamically_loaded_hipfftw& operator=(const dynamically_loaded_hipfftw&) = delete;
    dynamically_loaded_hipfftw& operator=(dynamically_loaded_hipfftw&&) = delete;

    static const dynamically_loaded_hipfftw& get_instance()
    {
        static dynamically_loaded_hipfftw singleton_instance;
        return singleton_instance;
    }

public:
    static LIB_HANDLE_T get_lib()
    {
        return get_instance().lib_handle;
    }
    static std::string get_load_error_info()
    {
        return get_instance().load_error_info.str();
    }
    ~dynamically_loaded_hipfftw()
    {
        if(lib_handle)
        {
#ifdef WIN32
            (void)FreeLibrary(lib_handle);
#else
            (void)dlclose(lib_handle);
#endif
        }
        lib_handle = nullptr;
    }
};

// exception to be caught at test execution, reporting issues caught when loading
// hipfftw and/or when fetching the address of the available hipfftw functions therefrom
struct hipfftw_undefined_function_ptr : std::runtime_error
{
    hipfftw_undefined_function_ptr(const std::string& info)
        : std::runtime_error(info)
    {
    }
};

// helper struct for retrieving a function's return type
template <class T>
struct func_ret;
template <typename R, class... Args>
struct func_ret<R(Args...)>
{
    using type = R;
};
template <class T>
using func_ret_t = typename func_ret<T>::type;

template <typename func_type, std::enable_if_t<std::is_function_v<func_type>, bool> = true>
struct dynamically_loaded_function_t
{
private:
    // address of the desired function, to be fetched from a dynamically loaded shared library
    func_type* func_ptr;
    // symbol of said function
    std::string func_symbol;

public:
    dynamically_loaded_function_t(const char* symbol)
        : func_ptr(nullptr)
        , func_symbol(symbol)
    {
    }

    // forwarding functional calls
    template <typename... Args>
    func_ret_t<func_type> operator()(Args... args) const
    {
        if(!may_be_used())
            throw hipfftw_undefined_function_ptr(dynamically_loaded_hipfftw::get_load_error_info());
        return func_ptr(args...);
    }
    void load_implementation()
    {
        const auto hipfftw_lib = dynamically_loaded_hipfftw::get_lib();
        if(!hipfftw_lib)
        {
            // make func_ptr unambiguously unset to force the dedicated exception
            // to be thrown at forwarded functional call(s)
            func_ptr = nullptr;
            return;
        }
#ifdef WIN32
        func_ptr = reinterpret_cast<func_type*>(GetProcAddress(hipfftw_lib, func_symbol.c_str()));
#else
        func_ptr = reinterpret_cast<func_type*>(dlsym(hipfftw_lib, func_symbol.c_str()));
#endif
    }
    bool may_be_used() const
    {
        return func_ptr != nullptr;
    }
};

template <typename T, typename... Args>
void load_implementations(dynamically_loaded_function_t<T>& first, Args&... others)
{
    first.load_implementation();
    if constexpr(sizeof...(others) > 0)
        load_implementations(others...);
}

// define singleton structures encapsulating the hipfftw function pointers
// (one specialization per supported precision)
template <fft_precision prec>
struct hipfftw_funcs;

#define HIPFFTW_STRINGIFY(x) #x
#define HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, func) \
    dynamically_loaded_function_t<decltype(prefix##func)> func            \
        = dynamically_loaded_function_t<decltype(prefix##func)>(HIPFFTW_STRINGIFY(prefix##func));

#define HIPFFTW_FUNCS_SPECIALIZATION(prefix, specialization)                         \
    template <>                                                                      \
    struct hipfftw_funcs<specialization>                                             \
    {                                                                                \
    private:                                                                         \
        hipfftw_funcs()                                                              \
        {                                                                            \
            load_implementations(malloc,                                             \
                                 alloc_real,                                         \
                                 alloc_complex,                                      \
                                 free,                                               \
                                 destroy_plan,                                       \
                                 cleanup,                                            \
                                 execute,                                            \
                                 plan_dft_1d,                                        \
                                 plan_dft_2d,                                        \
                                 plan_dft_3d,                                        \
                                 plan_dft,                                           \
                                 plan_dft_r2c_1d,                                    \
                                 plan_dft_r2c_2d,                                    \
                                 plan_dft_r2c_3d,                                    \
                                 plan_dft_r2c,                                       \
                                 plan_dft_c2r_1d,                                    \
                                 plan_dft_c2r_2d,                                    \
                                 plan_dft_c2r_3d,                                    \
                                 plan_dft_c2r,                                       \
                                 print_plan,                                         \
                                 set_timelimit,                                      \
                                 cost,                                               \
                                 flops);                                             \
        }                                                                            \
        /* disable copies and moves */                                               \
        hipfftw_funcs(const hipfftw_funcs&) = delete;                                \
        hipfftw_funcs& operator=(const hipfftw_funcs&) = delete;                     \
        hipfftw_funcs(hipfftw_funcs&&)                 = delete;                     \
        hipfftw_funcs& operator=(hipfftw_funcs&&) = delete;                          \
                                                                                     \
    public:                                                                          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, malloc)          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_real)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_complex)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, free)            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, destroy_plan)    \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cleanup)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_1d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_2d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_3d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_1d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_2d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_3d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c)    \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_1d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_2d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_3d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r)    \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, print_plan)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, set_timelimit)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cost)            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, flops)           \
        static const hipfftw_funcs& get_instance()                                   \
        {                                                                            \
            static const hipfftw_funcs instance;                                     \
            return instance;                                                         \
        }                                                                            \
    }

HIPFFTW_FUNCS_SPECIALIZATION(fftwf_, fft_precision_single);
HIPFFTW_FUNCS_SPECIALIZATION(fftw_, fft_precision_double);

enum class fftw_array_execute_type
{
    KNOWN, // for using plan creation's IO data at execution (plan needs to know said data at creation)
    NEW, // for updating the IO data at execution (plan needs to know said data at creation)
    UNSET, // default, lets the runtime decide what's best/required
};

// structure bundling a hipfftw plan and its (most recent) I/O data
template <fft_precision prec>
struct hipfftw_plan_bundle_t
{
private:
    const decltype(hipfftw_funcs<prec>::destroy_plan)& destroy_plan;

public:
    typename hipfftw_trait<prec>::plan_t plan;
    void*                                input; // nonowned
    void*                                output; // nonowned
    hipfftw_plan_bundle_t(decltype(destroy_plan) destroy_plan_impl)
        : destroy_plan(destroy_plan_impl)
        , plan(nullptr)
        , input(nullptr)
        , output(nullptr)
    {
    }
    ~hipfftw_plan_bundle_t()
    {
        // make this destructor non-throwing (e.g., function may not have been loaded successfully)
        if(destroy_plan.may_be_used())
            destroy_plan(plan);
    }
    // disable copies and moves
    hipfftw_plan_bundle_t(const hipfftw_plan_bundle_t&) = delete;
    hipfftw_plan_bundle_t& operator=(const hipfftw_plan_bundle_t&) = delete;
    hipfftw_plan_bundle_t(hipfftw_plan_bundle_t&&)                 = delete;
    hipfftw_plan_bundle_t& operator=(hipfftw_plan_bundle_t&&) = delete;
};

class hipfftw_params : public fft_params
{
private:
    // fftw provides multiple ways to create (combined allocate + init) FFT plans:
    // - plan_dft_Nd    : unbatched, default layout, dimension known at compile time (1, 2, or 3), length(s) representable as int
    // - plan_dft       : unbatched, default layout, dimension determined at run time, length(s) representable as int
    //
    // Rotate through the choices for better test coverage.
    enum class fftw_create_API
    {
        PLAN_DFT_ND,
        PLAN_DFT,
        NO_CAN_DO,
    };
    static constexpr unsigned default_fftw_flag = FFTW_ESTIMATE;

    // references to hipfftw implementaions
    const hipfftw_funcs<fft_precision_single>& hipfftwf_;
    const hipfftw_funcs<fft_precision_double>& hipfftw_;
    // declare possible hipfftw plans as shared pointers to enable safe
    // shallow copies for this class (gtest's ::testing::ValuesIn requires
    // copiable objects)
    std::shared_ptr<hipfftw_plan_bundle_t<fft_precision_single>> hipfftwf_plan_bundle;
    std::shared_ptr<hipfftw_plan_bundle_t<fft_precision_double>> hipfftw_plan_bundle;
    std::string                                                  current_token;
    // only supported case as of now: TODO, change this to UNSET once NEW is enabled
    fftw_array_execute_type preferred_execute_type = fftw_array_execute_type::KNOWN;

    bool fftw_basic_plan_can_do() const
    {
        return is_using_default_layout() && nbatch == 1
               && std::all_of(
                   length.begin(), length.end(), is_in_range<int, decltype(length)::value_type>);
    }

    // Return a suitable plan creation type for the current FFT parameters.
    fftw_create_API get_create_type() const
    {
        // check if hipfftw has the feature implemented
        if(is_planar())
            throw unimplemented_exception("planar I/O data are not supported by hipfftw");
        static_assert(std::is_same_v<decltype(ioffset), decltype(ooffset)>);
        const auto is_zero_offset_value
            = [](const decltype(ioffset)::value_type& i) { return i == 0; };

        if(precision == fft_precision_half || multiGPU > 1 || is_callback()
           || !std::all_of(ioffset.begin(), ioffset.end(), is_zero_offset_value)
           || !std::all_of(ooffset.begin(), ooffset.end(), is_zero_offset_value)
           || auto_allocate == fft_auto_allocation_off || mp_lib != fft_mp_lib_none
           || mp_comm != nullptr || !ifields.empty() || !ofields.empty() || scale_factor != 1.0)
        {
            // NO_CAN_DO for any feature that fftw itself does not enable:
            return fftw_create_API::NO_CAN_DO;
        }

        std::vector<fftw_create_API> allowed_apis;
        if(fftw_basic_plan_can_do())
        {
            if(0 < dim() && dim() <= 3)
                allowed_apis.push_back(fftw_create_API::PLAN_DFT_ND);
            allowed_apis.push_back(fftw_create_API::PLAN_DFT);
        }
        else
            throw unimplemented_exception(
                "parameters not supported by the current implementation of hipfftw");

        // hash the token to decide how to create this FFT.  we want
        // test cases to rotate between different create APIs, but we
        // also need the choice of API to be stable across reruns of
        // the same test cases.
        return allowed_apis[std::hash<std::string>()(token()) % allowed_apis.size()];
    }

    fft_status verify_placement(void* in, void* out) const
    {
        const auto placement_io = in == out ? fft_placement_inplace : fft_placement_notinplace;
        return placement == placement_io ? fft_status_success : fft_status_failure;
    }
    template <fft_precision prec>
    const hipfftw_funcs<prec>& get_hipfftw_implementation() const
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if constexpr(prec == fft_precision_single)
            return hipfftwf_;
        else
            return hipfftw_;
    }
    template <fft_precision prec>
    std::shared_ptr<hipfftw_plan_bundle_t<prec>>& get_hipfftw_plan_bundle()
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if constexpr(prec == fft_precision_single)
            return hipfftwf_plan_bundle;
        else
            return hipfftw_plan_bundle;
    }
    template <fft_precision prec>
    const std::shared_ptr<hipfftw_plan_bundle_t<prec>>& get_hipfftw_plan_bundle() const
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if constexpr(prec == fft_precision_single)
            return hipfftwf_plan_bundle;
        else
            return hipfftw_plan_bundle;
    }
    template <fft_precision prec>
    void set_hipfftw_plan_bundle(std::shared_ptr<hipfftw_plan_bundle_t<prec>>& plan_bundle_to_set)
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if constexpr(prec == fft_precision_single)
            hipfftwf_plan_bundle = plan_bundle_to_set;
        else
            hipfftw_plan_bundle = plan_bundle_to_set;
    }

    template <fft_precision prec>
    fft_status create_plan_dft_Nd(void* in, void* out)
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if(verify_placement(in, out) != fft_status_success)
            return fft_status_failure;

        const hipfftw_funcs<prec>& implementation = get_hipfftw_implementation<prec>();
        std::shared_ptr<hipfftw_plan_bundle_t<prec>> plan_bundle(
            new hipfftw_plan_bundle_t<prec>(implementation.destroy_plan));

        if(dim() == 1)
        {
            const int N = length[0];
            if(is_real())
            {
                if(is_forward())
                    plan_bundle->plan
                        = implementation.plan_dft_r2c_1d(N,
                                                         static_cast<hipfftw_real_t<prec>*>(in),
                                                         static_cast<hipfftw_complex_t<prec>*>(out),
                                                         default_fftw_flag);
                else
                    plan_bundle->plan
                        = implementation.plan_dft_c2r_1d(N,
                                                         static_cast<hipfftw_complex_t<prec>*>(in),
                                                         static_cast<hipfftw_real_t<prec>*>(out),
                                                         default_fftw_flag);
            }
            else
            {
                const int sign = is_forward() ? FFTW_FORWARD : FFTW_BACKWARD;
                plan_bundle->plan
                    = implementation.plan_dft_1d(N,
                                                 static_cast<hipfftw_complex_t<prec>*>(in),
                                                 static_cast<hipfftw_complex_t<prec>*>(out),
                                                 sign,
                                                 default_fftw_flag);
            }
        }
        else if(dim() == 2)
        {
            const int N0 = length[0];
            const int N1 = length[1];
            if(is_real())
            {
                if(is_forward())
                    plan_bundle->plan
                        = implementation.plan_dft_r2c_2d(N0,
                                                         N1,
                                                         static_cast<hipfftw_real_t<prec>*>(in),
                                                         static_cast<hipfftw_complex_t<prec>*>(out),
                                                         default_fftw_flag);
                else
                    plan_bundle->plan
                        = implementation.plan_dft_c2r_2d(N0,
                                                         N1,
                                                         static_cast<hipfftw_complex_t<prec>*>(in),
                                                         static_cast<hipfftw_real_t<prec>*>(out),
                                                         default_fftw_flag);
            }
            else
            {
                const int sign = is_forward() ? FFTW_FORWARD : FFTW_BACKWARD;
                plan_bundle->plan
                    = implementation.plan_dft_2d(N0,
                                                 N1,
                                                 static_cast<hipfftw_complex_t<prec>*>(in),
                                                 static_cast<hipfftw_complex_t<prec>*>(out),
                                                 sign,
                                                 default_fftw_flag);
            }
        }
        else
        {
            if(dim() != 3)
                throw logic_error("unexpected dimension detected  at plan creation");
            const int N0 = length[0];
            const int N1 = length[1];
            const int N2 = length[2];
            if(is_real())
            {
                if(is_forward())
                    plan_bundle->plan
                        = implementation.plan_dft_r2c_3d(N0,
                                                         N1,
                                                         N2,
                                                         static_cast<hipfftw_real_t<prec>*>(in),
                                                         static_cast<hipfftw_complex_t<prec>*>(out),
                                                         default_fftw_flag);
                else
                    plan_bundle->plan
                        = implementation.plan_dft_c2r_3d(N0,
                                                         N1,
                                                         N2,
                                                         static_cast<hipfftw_complex_t<prec>*>(in),
                                                         static_cast<hipfftw_real_t<prec>*>(out),
                                                         default_fftw_flag);
            }
            else
            {
                const int sign = is_forward() ? FFTW_FORWARD : FFTW_BACKWARD;
                plan_bundle->plan
                    = implementation.plan_dft_3d(N0,
                                                 N1,
                                                 N2,
                                                 static_cast<hipfftw_complex_t<prec>*>(in),
                                                 static_cast<hipfftw_complex_t<prec>*>(out),
                                                 sign,
                                                 default_fftw_flag);
            }
        }
        // I/O data
        plan_bundle->input  = in;
        plan_bundle->output = out;
        set_hipfftw_plan_bundle<prec>(plan_bundle);

        return fft_status_success;
    }

    template <fft_precision prec>
    fft_status create_plan_dft(void* in, void* out)
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if(verify_placement(in, out) != fft_status_success)
            return fft_status_failure;

        const hipfftw_funcs<prec>& implementation = get_hipfftw_implementation<prec>();
        std::shared_ptr<hipfftw_plan_bundle_t<prec>> plan_bundle(
            new hipfftw_plan_bundle_t<prec>(implementation.destroy_plan));

        int              rank = dim();
        std::vector<int> int_length(rank);
        std::transform(
            length.begin(),
            length.end(),
            int_length.begin(),
            [](const decltype(length)::value_type& val) { return static_cast<int>(val); });
        if(is_real())
        {
            if(is_forward())
                plan_bundle->plan
                    = implementation.plan_dft_r2c(rank,
                                                  int_length.data(),
                                                  static_cast<hipfftw_real_t<prec>*>(in),
                                                  static_cast<hipfftw_complex_t<prec>*>(out),
                                                  default_fftw_flag);
            else
                plan_bundle->plan
                    = implementation.plan_dft_c2r(rank,
                                                  int_length.data(),
                                                  static_cast<hipfftw_complex_t<prec>*>(in),
                                                  static_cast<hipfftw_real_t<prec>*>(out),
                                                  default_fftw_flag);
        }
        else
        {
            const int sign    = is_forward() ? FFTW_FORWARD : FFTW_BACKWARD;
            plan_bundle->plan = implementation.plan_dft(rank,
                                                        int_length.data(),
                                                        static_cast<hipfftw_complex_t<prec>*>(in),
                                                        static_cast<hipfftw_complex_t<prec>*>(out),
                                                        sign,
                                                        default_fftw_flag);
        }
        plan_bundle->input  = in;
        plan_bundle->output = out;
        set_hipfftw_plan_bundle<prec>(plan_bundle);
        return fft_status_success;
    }

    // check that the object does know of a hipfftw plan of the desired precision
    // and that the current token matches expectations
    template <fft_precision prec>
    bool plan_is_valid() const
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        const auto plan_bundle = get_hipfftw_plan_bundle<prec>();
        return plan_bundle && plan_bundle->plan && current_token == token();
    }

    template <fft_precision prec>
    fft_status internal_create_plan(void* in, void* out)
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if(prec != precision)
            throw logic_error("parameter's precision does not match invoked specialization for "
                              "internal plan creation");
        // check if we need to create a new plan
        if(plan_is_valid<prec>())
        {
            return fft_status_success;
        }

        internal_free<prec>();
        fft_status ret{fft_status_failure};
        switch(get_create_type())
        {
        case fftw_create_API::PLAN_DFT_ND:
        {
            ret = create_plan_dft_Nd<prec>(in, out);
            break;
        }
        case fftw_create_API::PLAN_DFT:
        {
            ret = create_plan_dft<prec>(in, out);
            break;
        }
        case fftw_create_API::NO_CAN_DO:
        {
            ret = fft_status_failure;
            break;
        }
        default:
        {
            throw logic_error("Unexpected plan creation type");
        }
        }

        if(ret != fft_status_success)
            return ret;
        // store token to reliably check if plan was already created thereafter
        current_token = token();
        return plan_is_valid<prec>() ? fft_status_success : fft_status_failure;
    }

    template <fft_precision prec>
    fft_status internal_execute(void* in, void* out)
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        if(!plan_is_valid<prec>() && internal_create_plan<prec>(in, out) != fft_status_success)
            return fft_status_failure;
        auto plan_bundle = get_hipfftw_plan_bundle<prec>();
        switch(preferred_execute_type)
        {
        case fftw_array_execute_type::KNOWN:
            if(in == plan_bundle->input && out == plan_bundle->output)
            {
                get_hipfftw_implementation<prec>().execute(plan_bundle->plan);
                break;
            }
            // the preferred_execute_type couldn't be honored, we can't ignored the I/O data
            [[fallthrough]];
        case fftw_array_execute_type::NEW:
            throw unimplemented_exception(
                "execution not implemented yet for new I/O data at execution");
            break;
        case fftw_array_execute_type::UNSET: /* cannot happen at execution for any valid plan */
        default:
            throw logic_error("invalid array execution type encountered at execution");
            break;
        }
        // NOTE: given the signature of fftw{f}_execute*, this call will always
        // "succeed" if it reaches this point...
        return fft_status_success;
    }

    template <fft_precision prec>
    void internal_free()
    {
        static_assert(prec == fft_precision_single || prec == fft_precision_double);
        get_hipfftw_plan_bundle<prec>().reset();
    }

public:
    hipfftw_params()
        : hipfftwf_(hipfftw_funcs<fft_precision_single>::get_instance())
        , hipfftw_(hipfftw_funcs<fft_precision_double>::get_instance())
        , hipfftwf_plan_bundle(nullptr)
        , hipfftw_plan_bundle(nullptr)
    {
    }

    ~hipfftw_params() override
    {
        free();
    }

    fft_status set_callbacks(void*  load_cb_host,
                             void*  load_cb_data,
                             void*  store_cb_host,
                             void*  store_cb_data,
                             size_t load_cb_shared_mem_bytes  = 0,
                             size_t store_cb_shared_mem_bytes = 0) override
    {
        return fft_status_failure;
    }

    virtual fft_status execute(void** in, void** out) override
    {
        return execute(in[0], out[0]);
    }

    fft_status execute(void* in, void* out)
    {
        switch(precision)
        {
        case fft_precision_single:
            return internal_execute<fft_precision_single>(in, out);
            break;
        case fft_precision_double:
            return internal_execute<fft_precision_double>(in, out);
            break;
        case fft_precision_half:
            throw unimplemented_exception("precision is not supported by hipfftw (nor by fftw)");
            break;
        default:
            throw logic_error("unknown precision encountered at execution");
            break;
        }
    }

    // plan creation unaware of data pointers:
    fft_status create_plan() override
    {
        // this plan creation is unaware of I/O data,
        // the execution I/O must be accounted for by default
        if(preferred_execute_type == fftw_array_execute_type::UNSET)
            set_preferred_execution_type(fftw_array_execute_type::NEW);
        fft_status ret{fft_status_failure};
        switch(preferred_execute_type)
        {
        case fftw_array_execute_type::KNOWN:
            // we must know the array at plan creation, we can't create the plan here
            // --> do it on-the-fly when execute is invoked
            free();
            ret = fft_status_success;
            break;
        case fftw_array_execute_type::NEW:
            // TODO: create temp I/O data ptr, call create_plan(temp_in, temp_out);
            break;
        case fftw_array_execute_type::UNSET:
        default:
            throw logic_error("invalid array execution type encountered at plan creation");
            break;
        }
        return ret;
    }

    // plan creation aware of data pointers:
    fft_status create_plan(void* in, void* out)
    {
        if(preferred_execute_type == fftw_array_execute_type::UNSET)
            set_preferred_execution_type(fftw_array_execute_type::NEW);
        fft_status ret{fft_status_failure};
        switch(precision)
        {
        case fft_precision_single:
            ret = internal_create_plan<fft_precision_single>(in, out);
            break;
        case fft_precision_double:
            ret = internal_create_plan<fft_precision_double>(in, out);
            break;
        case fft_precision_half:
            throw unimplemented_exception("precision is not supported by hipfftw (nor by fftw)");
            break;
        default:
            throw logic_error("unknown precision encountered at plan creation");
            break;
        }
        return ret;
    }

    void free()
    {
        internal_free<fft_precision_single>();
        internal_free<fft_precision_double>();
    }

    void set_preferred_execution_type(fftw_array_execute_type type_to_set)
    {
        preferred_execute_type = type_to_set;
    }

    struct logic_error : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
};

#endif
