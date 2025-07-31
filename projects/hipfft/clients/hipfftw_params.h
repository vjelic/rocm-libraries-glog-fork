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

#include "../shared/fft_params.h"
#include <exception>
#include <fftw3.h>
#include <string>

#ifdef WIN32
#include <windows.h>
// psapi.h requires windows.h to be included first
#include <psapi.h>
#else
#include <dlfcn.h>
#include <link.h>
#endif

#define HIPFFTW_API_WRAP(prefix, func)            \
    decltype(&prefix##func) func       = nullptr; \
    decltype(&prefix##func) ref_##func = &prefix##func
#define HIPFFTW_API_LOAD(prefix, func) \
    func = reinterpret_cast<decltype(&prefix##func)>(lib_symbol(lib, #prefix #func));

#define HIPFFTW_API_WRAP_ALL(prefix)           \
    HIPFFTW_API_WRAP(prefix, malloc);          \
    HIPFFTW_API_WRAP(prefix, free);            \
    HIPFFTW_API_WRAP(prefix, destroy_plan);    \
    HIPFFTW_API_WRAP(prefix, cleanup);         \
    HIPFFTW_API_WRAP(prefix, execute);         \
    HIPFFTW_API_WRAP(prefix, plan_dft_1d);     \
    HIPFFTW_API_WRAP(prefix, plan_dft_2d);     \
    HIPFFTW_API_WRAP(prefix, plan_dft_3d);     \
    HIPFFTW_API_WRAP(prefix, plan_dft);        \
    HIPFFTW_API_WRAP(prefix, plan_dft_r2c_1d); \
    HIPFFTW_API_WRAP(prefix, plan_dft_r2c_2d); \
    HIPFFTW_API_WRAP(prefix, plan_dft_r2c_3d); \
    HIPFFTW_API_WRAP(prefix, plan_dft_r2c);    \
    HIPFFTW_API_WRAP(prefix, plan_dft_c2r_1d); \
    HIPFFTW_API_WRAP(prefix, plan_dft_c2r_2d); \
    HIPFFTW_API_WRAP(prefix, plan_dft_c2r_3d); \
    HIPFFTW_API_WRAP(prefix, plan_dft_c2r);    \
    HIPFFTW_API_WRAP(prefix, print_plan);      \
    HIPFFTW_API_WRAP(prefix, set_timelimit);   \
    HIPFFTW_API_WRAP(prefix, cost);            \
    HIPFFTW_API_WRAP(prefix, flops);

#define HIPFFTW_API_LOAD_ALL(prefix)           \
    HIPFFTW_API_LOAD(prefix, malloc);          \
    HIPFFTW_API_LOAD(prefix, free);            \
    HIPFFTW_API_LOAD(prefix, destroy_plan);    \
    HIPFFTW_API_LOAD(prefix, cleanup);         \
    HIPFFTW_API_LOAD(prefix, execute);         \
    HIPFFTW_API_LOAD(prefix, plan_dft_1d);     \
    HIPFFTW_API_LOAD(prefix, plan_dft_2d);     \
    HIPFFTW_API_LOAD(prefix, plan_dft_3d);     \
    HIPFFTW_API_LOAD(prefix, plan_dft);        \
    HIPFFTW_API_LOAD(prefix, plan_dft_r2c_1d); \
    HIPFFTW_API_LOAD(prefix, plan_dft_r2c_2d); \
    HIPFFTW_API_LOAD(prefix, plan_dft_r2c_3d); \
    HIPFFTW_API_LOAD(prefix, plan_dft_r2c);    \
    HIPFFTW_API_LOAD(prefix, plan_dft_c2r_1d); \
    HIPFFTW_API_LOAD(prefix, plan_dft_c2r_2d); \
    HIPFFTW_API_LOAD(prefix, plan_dft_c2r_3d); \
    HIPFFTW_API_LOAD(prefix, plan_dft_c2r);    \
    HIPFFTW_API_LOAD(prefix, print_plan);      \
    HIPFFTW_API_LOAD(prefix, set_timelimit);   \
    HIPFFTW_API_LOAD(prefix, cost);            \
    HIPFFTW_API_LOAD(prefix, flops);

struct hipfftw_funcs_base
{
    hipfftw_funcs_base()
    {
        // open the hipfftw library
        // FIXME: needs to be hipfftw.dll on Windows
        lib = lib_load("libhipfftw.so");
        if(!lib)
            // FIXME: maybe catch this in tests, to skip instead of fail?
            throw std::runtime_error(lib_load_error());
    }
    ~hipfftw_funcs_base()
    {
        lib_close(lib);
    }
    hipfftw_funcs_base(const hipfftw_funcs_base&) = delete;
    hipfftw_funcs_base& operator=(const hipfftw_funcs_base&) = delete;

#ifdef WIN32
    typedef HMODULE HIPFFTW_LIB;
#else
    typedef void* HIPFFTW_LIB;
#endif

    // Load the hipfftw library
    static HIPFFTW_LIB lib_load(const std::string& path)
    {
#ifdef WIN32
        return LoadLibraryA(path.c_str());
#else
        return dlopen(path.c_str(), RTLD_LAZY);
#endif
    }

    // Return a string describing the error loading hipfftw
    static const char* lib_load_error()
    {
#ifdef WIN32
        // just return the error number
        static std::string error_str;
        error_str = std::to_string(GetLastError());
        return error_str.c_str();
#else
        return dlerror();
#endif
    }

    // Get symbol from hipfftw lib
    static void* lib_symbol(HIPFFTW_LIB libhandle, const char* sym)
    {
#ifdef WIN32
        return reinterpret_cast<void*>(GetProcAddress(libhandle, sym));
#else
        return dlsym(libhandle, sym);
#endif
    }

    static void lib_close(HIPFFTW_LIB libhandle)
    {
        if(!libhandle)
            return;
#ifdef WIN32
        FreeLibrary(libhandle);
#else
        dlclose(libhandle);
#endif
    }

    HIPFFTW_LIB lib = nullptr;
};

struct hipfftw_funcs : public hipfftw_funcs_base
{
    hipfftw_funcs()
    {
        HIPFFTW_API_LOAD_ALL(fftw_);
    }
    HIPFFTW_API_WRAP_ALL(fftw_);

    static const fft_precision precision = fft_precision_double;
    typedef double             Tfloat;
    typedef fftw_complex       Tcomplex;
};

struct hipfftwf_funcs : public hipfftw_funcs_base
{
    hipfftwf_funcs()
    {
        HIPFFTW_API_LOAD_ALL(fftwf_);
    }
    HIPFFTW_API_WRAP_ALL(fftwf_);

    static const fft_precision precision = fft_precision_single;
    typedef float              Tfloat;
    typedef fftwf_complex      Tcomplex;
};

class hipfftw_params : public fft_params
{
public:
    hipfftw_params()           = default;
    ~hipfftw_params() override = default;

    fft_status set_callbacks(void* load_cb_host,
                             void* load_cb_data,
                             void* store_cb_host,
                             void* store_cb_data) override
    {
        throw std::runtime_error("callbacks are not supported in FFTW");
    }

    fft_status execute(void** in, void** out) override
    {
        return fft_status_failure;
    }

    fft_status create_plan() override
    {
        return fft_status_failure;
    }

private:
};

#endif
