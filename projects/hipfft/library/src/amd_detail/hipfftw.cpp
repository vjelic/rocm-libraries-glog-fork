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
#include "../../../shared/ptrdiff.h"
#include "rocfft/rocfft.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

// anonymous namespace for implementation details
namespace
{
    struct hipfftw_invalid_arg : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_unsupported : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_invalid_plan : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_internal_error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct rocfft_failure : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_bad_alloc : public std::runtime_error
    {
        const size_t attempted_size;
        hipfftw_bad_alloc(const std::string& info, size_t alloc_size)
            : std::runtime_error::runtime_error(info)
            , attempted_size(alloc_size)
        {
        }
    };
    struct hipfftw_bad_gpu_alloc : public hipfftw_bad_alloc
    {
        using hipfftw_bad_alloc::hipfftw_bad_alloc;
    };
    struct hipfftw_execution_error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    struct hipfftw_flow_redirection : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    enum class hipfftw_io_label
    {
        IN,
        OUT,
    };

    constexpr bool is_real(rocfft_transform_type dft_type)
    {
        return dft_type == rocfft_transform_type_real_forward
               || dft_type == rocfft_transform_type_real_inverse;
    }

    template <rocfft_precision prec>
    struct hipfftw_scalar_trait;
    template <>
    struct hipfftw_scalar_trait<rocfft_precision_single>
    {
        using complex_t = fftwf_complex;
        using real_t    = float;
    };
    template <>
    struct hipfftw_scalar_trait<rocfft_precision_double>
    {
        using complex_t = fftw_complex;
        using real_t    = double;
    };

    template <rocfft_precision prec>
    using hipfftw_complex_data_t = typename hipfftw_scalar_trait<prec>::complex_t;
    template <rocfft_precision prec>
    using hipfftw_real_data_t = typename hipfftw_scalar_trait<prec>::real_t;
    // template helper struct for data type consistency (compile-time checks)
    template <rocfft_transform_type dft_type, rocfft_precision prec, hipfftw_io_label io>
    using hipfftw_user_data_t
        = std::conditional_t<!is_real(dft_type)
                                 || (dft_type == rocfft_transform_type_real_forward
                                     ^ io == hipfftw_io_label::IN),
                             // user data is complex
                             hipfftw_complex_data_t<prec>,
                             // user data is real
                             hipfftw_real_data_t<prec>>;

    template <rocfft_transform_type dft_type, hipfftw_io_label io>
    constexpr rocfft_array_type hipfftw_get_array_type()
    {
        if constexpr(!is_real(dft_type))
            return rocfft_array_type_complex_interleaved;
        else if constexpr(dft_type == rocfft_transform_type_real_forward
                          ^ io == hipfftw_io_label::IN)
            return rocfft_array_type_hermitian_interleaved;
        else
            return rocfft_array_type_real;
    }

    struct hipfftw_raii_recording_event
    {
    private:
        hipEvent_t ev;

    public:
        hipfftw_raii_recording_event()
        {
            if(hipEventCreate(&ev) != hipSuccess)
            {
                throw std::runtime_error("failed to create an event");
            }
            if(hipEventRecord(ev) != hipSuccess)
            {
                throw std::runtime_error("failed to record an event");
            }
        }
        // disable copies and moves
        hipfftw_raii_recording_event(hipfftw_raii_recording_event&& other) = delete;
        hipfftw_raii_recording_event& operator=(hipfftw_raii_recording_event&& other) = delete;
        hipfftw_raii_recording_event(const hipfftw_raii_recording_event& other)       = delete;
        hipfftw_raii_recording_event& operator=(const hipfftw_raii_recording_event& other) = delete;

        ~hipfftw_raii_recording_event()
        {
            synchronize();
            (void)hipEventDestroy(ev);
        }
        void synchronize()
        {
            if(hipEventSynchronize(ev) != hipSuccess)
            {
                throw std::runtime_error("failed to synchronize on an event");
            }
        }
    };

    enum class hipfftw_memcpy_kind : std::underlying_type_t<hipMemcpyKind>
    {
        H2D = static_cast<std::underlying_type_t<hipMemcpyKind>>(hipMemcpyHostToDevice),
        D2H = static_cast<std::underlying_type_t<hipMemcpyKind>>(hipMemcpyDeviceToHost),
        D2D = static_cast<std::underlying_type_t<hipMemcpyKind>>(hipMemcpyDeviceToDevice),
        NONE,
    };
    enum class hipfftw_memcpy_direction
    {
        TO,
        FROM,
    };

    // constexpr used for readability
    constexpr bool hipfftw_owns_it = true;

    template <bool owning>
    struct hipfftw_data_ptr_bundle
    {
    private:
        void*                 ptr;
        hipPointerAttribute_t attributes;

        void free_owned_resources()
        {
            if constexpr(!owning)
                return;
            if(!ptr)
                return;
            switch(attributes.type)
            {
            case hipMemoryType::hipMemoryTypeManaged:
            case hipMemoryType::hipMemoryTypeDevice:
                (void)hipFree(ptr);
                break;
            case hipMemoryType::hipMemoryTypeHost:
                (void)hipHostFree(ptr);
                break;
            case hipMemoryType::hipMemoryTypeUnregistered:
                free(ptr);
                break;
            // hipMemoryTypeUnified & hipMemoryTypeArray not set by hipPointerGetAttributes on AMD platforms
            case hipMemoryType::hipMemoryTypeArray:
            case hipMemoryType::hipMemoryTypeUnified:
            default:
                throw hipfftw_internal_error("unexpected type of (owned) allocation ancountered by "
                                             "hipfftw_data_ptr_bundle::free_owned_resources");
                break;
            }
            ptr = nullptr;
        }

    public:
        hipfftw_data_ptr_bundle(void* init_ptr = nullptr)
            : ptr(nullptr)
        {
            set(init_ptr);
        }
        void set(void* new_ptr)
        {
            if(ptr == new_ptr)
                return;
            free_owned_resources();
            ptr = new_ptr;
            // hipPointerGetAttributes sets default attributes
            // (with attributes.type == hipMemoryTypeUnregistered) if
            // new_ptr = nullptr or if new_ptr is not to be found in the map
            // that the run-time manages (error possibly reported to a log in
            // the latter case in case of debug runtime version)
            if(hipPointerGetAttributes(&attributes, new_ptr) != hipSuccess)
            {
                throw std::runtime_error(
                    "hipfftw_data_ptr_bundle::set failed to determine pointer attributes");
            }
        }
        ~hipfftw_data_ptr_bundle()
        {
            free_owned_resources();
        }

        template <hipfftw_memcpy_direction dir>
        hipfftw_memcpy_kind get_copy_kind(int deviceId) const
        {
            static_assert(dir == hipfftw_memcpy_direction::TO
                          || dir == hipfftw_memcpy_direction::FROM);
            switch(attributes.type)
            {
            case hipMemoryType::hipMemoryTypeManaged:
                return hipfftw_memcpy_kind::NONE; // the runtime is supposed to manage it
            case hipMemoryType::hipMemoryTypeDevice:
                return attributes.device != deviceId ? hipfftw_memcpy_kind::D2D
                                                     : hipfftw_memcpy_kind::NONE;
            case hipMemoryType::hipMemoryTypeUnregistered:
            case hipMemoryType::hipMemoryTypeHost:
                // TODO: check if device is APU and return false (systematically?) in that case
                if constexpr(dir == hipfftw_memcpy_direction::TO)
                    return hipfftw_memcpy_kind::H2D;
                else
                    return hipfftw_memcpy_kind::D2H;
            // hipMemoryTypeUnified & hipMemoryTypeArray not set by hipPointerGetAttributes on AMD platforms
            case hipMemoryType::hipMemoryTypeArray:
            case hipMemoryType::hipMemoryTypeUnified:
            default:
                throw hipfftw_internal_error("unexpected type of memory encountered by "
                                             "hipfftw_data_ptr_bundle::get_copy_kind");
            }
            // unreachable
        }

        void* get_data_ptr() const
        {
            return ptr;
        }

        bool operator==(const hipfftw_data_ptr_bundle& other) const
        {
            return ptr == other.ptr;
        }
        bool operator!=(const hipfftw_data_ptr_bundle& other) const
        {
            return !(*this == other);
        }

        // escalate implicit conversions to bool:
        operator bool() const
        {
            return ptr;
        }

        // disable copies and move
        hipfftw_data_ptr_bundle(const hipfftw_data_ptr_bundle&) = delete;
        hipfftw_data_ptr_bundle& operator=(const hipfftw_data_ptr_bundle&) = delete;
        hipfftw_data_ptr_bundle(hipfftw_data_ptr_bundle&&)                 = delete;
        hipfftw_data_ptr_bundle& operator=(hipfftw_data_ptr_bundle&&) = delete;
    };

    int hipfftw_get_current_device_id()
    {
        auto ret = hipInvalidDeviceId;
        if(hipGetDevice(&ret) != hipSuccess || ret == hipInvalidDeviceId)
            throw std::runtime_error("device id could not be successfully determined");
        return ret;
    }

    // helper routine for assigning a device allocation to an owning data_ptr_bundle
    void hipfftw_set_device_allocation_for(hipfftw_data_ptr_bundle<hipfftw_owns_it>& bundle,
                                           size_t                                    alloc_size,
                                           const std::string&                        buffer_label,
                                           const int& desired_device_id)
    {
        void* temp = nullptr;
        if(alloc_size > 0)
        {
            std::ostringstream info;
            const auto         current_device_id = hipfftw_get_current_device_id();
            // set device id to the desired one
            if(current_device_id != desired_device_id
               && hipSetDevice(desired_device_id) != hipSuccess)
            {
                info << "failed to temporarily set device id to " << desired_device_id
                     << " when creating " << buffer_label;
                throw std::runtime_error(info.str());
            }
            if(hipMalloc(&temp, alloc_size) != hipSuccess || !temp)
            {
                info << "device allocation for " << buffer_label << " failed";
                throw hipfftw_bad_gpu_alloc(info.str(), alloc_size);
            }
            // reset device id to what it was
            if(current_device_id != desired_device_id
               && hipSetDevice(current_device_id) != hipSuccess)
            {
                info << "failed to reset device id to " << current_device_id << " upon creation of "
                     << buffer_label;
                throw std::runtime_error(info.str());
            }
        }
        bundle.set(temp);
    }

    struct hipfftw_plan_internal
    {
        hipfftw_plan_internal() = default;
        ~hipfftw_plan_internal()
        {
            if(internal_rocfft_info)
            {
                rocfft_execution_info_destroy(internal_rocfft_info);
                internal_rocfft_info = nullptr;
            }
            if(internal_rocfft_desc)
            {
                rocfft_plan_description_destroy(internal_rocfft_desc);
                internal_rocfft_desc = nullptr;
            }
            if(internal_rocfft_plan)
            {
                rocfft_plan_destroy(internal_rocfft_plan);
                internal_rocfft_plan = nullptr;
            }
        }

        // disallow copies and moves
        hipfftw_plan_internal(const hipfftw_plan_internal&) = delete;
        hipfftw_plan_internal& operator=(const hipfftw_plan_internal&) = delete;
        hipfftw_plan_internal(hipfftw_plan_internal&&)                 = delete;
        hipfftw_plan_internal& operator=(hipfftw_plan_internal&&) = delete;

        rocfft_plan             internal_rocfft_plan = nullptr;
        rocfft_plan_description internal_rocfft_desc = nullptr;
        rocfft_execution_info   internal_rocfft_info = nullptr;
        rocfft_result_placement plan_placement;
        // Sizes of the buffers so we know how much to copy
        size_t in_bytes         = 0;
        size_t out_bytes        = 0;
        size_t work_buffer_size = 0;
        // Once initialized, the plan is configured for the device id set when initializing it
        int device_id = hipInvalidDeviceId;
        // bundles for owned data pointers
        hipfftw_data_ptr_bundle<hipfftw_owns_it> work_buffer;
        // possibly modified in new-array execute paths
        mutable hipfftw_data_ptr_bundle<hipfftw_owns_it> in_device;
        mutable hipfftw_data_ptr_bundle<hipfftw_owns_it> out_device;
        // bundles for non-owned (user's) data pointers, possibly modified in new-array execute paths
        mutable hipfftw_data_ptr_bundle<!hipfftw_owns_it> input;
        mutable hipfftw_data_ptr_bundle<!hipfftw_owns_it> output;

        void execute() const
        {
            if(!internal_rocfft_plan)
                throw hipfftw_invalid_plan("uninitialized plan used at execution");
            if(!input || !output)
                throw hipfftw_invalid_arg("invalid data pointers for execution (nullptr's)");
            // in/out may or may not need to be copied to the device
            const auto input_copy_kind
                = input.get_copy_kind<hipfftw_memcpy_direction::TO>(device_id);
            const auto output_copy_kind
                = output.get_copy_kind<hipfftw_memcpy_direction::FROM>(device_id);

            void* in_exec = input.get_data_ptr();
            if(input_copy_kind != hipfftw_memcpy_kind::NONE)
            {
                if(hipMemcpyAsync(in_exec,
                                  in_device.get_data_ptr(),
                                  in_bytes,
                                  static_cast<hipMemcpyKind>(input_copy_kind))
                   != hipSuccess)
                {
                    throw hipfftw_execution_error(
                        "failed to copy input data into GPU input buffer");
                }
                in_exec = in_device.get_data_ptr();
            }
            void* out_exec
                = plan_placement == rocfft_placement_inplace
                      ? in_exec
                      : (output_copy_kind != hipfftw_memcpy_kind::NONE ? out_device.get_data_ptr()
                                                                       : output.get_data_ptr());
            rocfft_execute(internal_rocfft_plan, &in_exec, &out_exec, internal_rocfft_info);
            if(output_copy_kind != hipfftw_memcpy_kind::NONE)
            {
                if(hipMemcpyAsync(output.get_data_ptr(),
                                  out_exec,
                                  out_bytes,
                                  static_cast<hipMemcpyKind>(output_copy_kind))
                   != hipSuccess)
                {
                    throw hipfftw_execution_error(
                        "failed to copy output data from GPU output buffer");
                }
                if(output_copy_kind == hipfftw_memcpy_kind::D2H)
                {
                    // we must assume the data may be accessed on the host immediately after completion
                    // of this call: the following enforces synchronization as it goes out of scope.
                    hipfftw_raii_recording_event ev;
                }
            }
            return;
        }

        void init_rocfft() const
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
        }

        // NOTE: const-qualified routine modifying mutable members
        // Motivation: new-array execute functions do require const qualifier
        // yet it may need the input/output bundles to be updated and may need
        // to create the I/O device buffers
        void update_io_ptr_and_buffers(void* user_in, void* user_out) const
        {
            // cannot be called before internal structures were created
            if(!internal_rocfft_info || !internal_rocfft_desc || !internal_rocfft_plan)
                throw hipfftw_internal_error("unexpected call to update_io_ptr");
            // check that placement requirement is honored by the new user I/O
            // if internal_rocfft_plan already exists
            const auto new_io_placement
                = user_in == user_out ? rocfft_placement_inplace : rocfft_placement_notinplace;
            if(new_io_placement != plan_placement)
                throw hipfftw_invalid_arg(
                    "the new arrays do not honor the plan placement requirement");
            // possibly modifying mutable, below
            input.set(user_in);
            output.set(user_out);

            // set device I/O buffer, if they're needed
            // TODO: check if the current device is an APU and simply never use
            // I/O device buffers in that case
            if(input.get_copy_kind<hipfftw_memcpy_direction::TO>(device_id)
               != hipfftw_memcpy_kind::NONE)
            {
                if(!in_device)
                    hipfftw_set_device_allocation_for(
                        in_device, in_bytes, "input buffer", device_id);
            }
            else // in_device is not needed, free resources to minimize device memory footprint
                in_device.set(nullptr);

            if(input != output
               && output.get_copy_kind<hipfftw_memcpy_direction::FROM>(device_id)
                      != hipfftw_memcpy_kind::NONE)
            {
                if(!out_device)
                    hipfftw_set_device_allocation_for(
                        out_device, out_bytes, "output buffer", device_id);
            }
            else // out_deviceis not needed, free resources to minimize device memory footprint
                out_device.set(nullptr);

            return;
        }

        template <rocfft_transform_type dft_type,
                  rocfft_precision      prec,
                  size_t                rank,
                  typename T,
                  size_t batch_rank>
        void init(const std::array<T, rank>&                                  lengths_rm,
                  const std::array<T, rank>&                                  istrides_rm,
                  const std::array<T, rank>&                                  ostrides_rm,
                  hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::IN>*  user_in,
                  hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUT>* user_out,
                  const std::array<T, batch_rank>&                            batch,
                  const std::array<T, batch_rank>&                            idist,
                  const std::array<T, batch_rank>&                            odist,
                  unsigned                                                    flags)
        {
            // compile-time validations of template specialization values
            static_assert(1 <= rank && rank <= 3);
            static_assert(1 == batch_rank); // only supported case at the moment
            // no support for half precision
            static_assert(prec == rocfft_precision_single || prec == rocfft_precision_double);
            // assuming no overflow when converting values from T into size_t below
            static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<size_t>::max());
            // Validation of input arguments:
            auto is_strictly_negative = [](const T& val) { return val < static_cast<T>(0); };
            auto is_strictly_positive = [](const T& val) { return val > static_cast<T>(0); };
            if(!std::all_of(lengths_rm.begin(), lengths_rm.end(), is_strictly_positive))
                throw hipfftw_invalid_arg("length(s) must be strictly positive");
            // Negative strides are not supported
            for(const auto& strides : {istrides_rm, ostrides_rm})
                if(std::any_of(strides.begin(), strides.end(), is_strictly_negative))
                    throw hipfftw_unsupported("strides must not be negative");
            if(!std::all_of(batch.begin(), batch.end(), is_strictly_positive))
                throw hipfftw_invalid_arg("batch(es) must be strictly positive");
            // Negative distances are not supported
            for(const auto& dist : {idist, odist})
                if(std::any_of(dist.begin(), dist.end(), is_strictly_negative))
                    throw hipfftw_unsupported("distances must not be negative");
            // Valid flag values are defined as bitwise OR of zero or more (unsigned) power-of-2
            // compile-time constants, (enabling well-defined identification via bitwise manipulations).
            if(flags
               & ~(FFTW_WISDOM_ONLY | FFTW_MEASURE | FFTW_DESTROY_INPUT | FFTW_UNALIGNED
                   | FFTW_CONSERVE_MEMORY | FFTW_EXHAUSTIVE | FFTW_PRESERVE_INPUT | FFTW_PATIENT
                   | FFTW_ESTIMATE))
            {
                throw hipfftw_invalid_arg("invalid flag");
            }
            if((!user_in || !user_out) && !(flags & FFTW_ESTIMATE) && !(flags & FFTW_WISDOM_ONLY))
            {
                throw hipfftw_invalid_arg("nullptr input/output is invalid with given flags");
            }

            if(flags & FFTW_WISDOM_ONLY)
            {
                throw hipfftw_unsupported("usage of FFTW_WISDOM_ONLY is not supported");
            }

            if constexpr(dft_type == rocfft_transform_type_real_inverse && rank > 1)
            {
                if(flags & FFTW_PRESERVE_INPUT)
                {
                    throw hipfftw_unsupported(
                        "input preservation is not supported for multi-dimensional C2R");
                }
            }
            // Generalized input are validated... Let's initialize the plan!
            init_rocfft(); // "magic" common to all template specializations

            // Change row-major to col-major for all relevant inputs, converting from T to size_t
            auto reverse = [](const std::array<T, rank>& in_array) {
                auto ret = std::array<size_t, rank>();
                std::reverse_copy(in_array.begin(), in_array.end(), ret.begin());
                return ret;
            };
            const auto lengths_cm  = reverse(lengths_rm);
            const auto istrides_cm = reverse(istrides_rm);
            const auto ostrides_cm = reverse(ostrides_rm);

            // Create plan description if necessary
            if(rocfft_plan_description_create(&internal_rocfft_desc) != rocfft_status_success)
            {
                throw rocfft_failure("rocfft failed to create a plan description");
            }
            if(rocfft_plan_description_set_data_layout(
                   internal_rocfft_desc,
                   hipfftw_get_array_type<dft_type, hipfftw_io_label::IN>(),
                   hipfftw_get_array_type<dft_type, hipfftw_io_label::OUT>(),
                   nullptr,
                   nullptr,
                   rank,
                   istrides_cm.data(),
                   idist[0],
                   rank,
                   ostrides_cm.data(),
                   odist[0])
               != rocfft_status_success)
            {
                throw rocfft_failure("rocfft failed to set a data layout");
            }

            plan_placement = static_cast<void*>(user_in) == static_cast<void*>(user_out)
                                 ? rocfft_placement_inplace
                                 : rocfft_placement_notinplace;
            if(rocfft_plan_create(&internal_rocfft_plan,
                                  plan_placement,
                                  dft_type,
                                  prec,
                                  rank,
                                  lengths_cm.data(),
                                  batch[0],
                                  internal_rocfft_desc)
               != rocfft_status_success)
            {
                throw rocfft_failure("rocfft failed to create a plan");
            }

            if(rocfft_execution_info_create(&internal_rocfft_info) != rocfft_status_success)
            {
                throw rocfft_failure("rocfft failed to create an info struct");
            }
            if(rocfft_plan_get_work_buffer_size(internal_rocfft_plan, &work_buffer_size)
               != rocfft_status_success)
            {
                throw rocfft_failure("rocfft failed to fetch size of work area");
            }
            // set device id
            device_id = hipfftw_get_current_device_id();
            // create and set device work buffer
            if(work_buffer_size > 0)
            {
                hipfftw_set_device_allocation_for(
                    work_buffer, work_buffer_size, "work buffer", device_id);
                if(rocfft_execution_info_set_work_buffer(
                       internal_rocfft_info, work_buffer.get_data_ptr(), work_buffer_size)
                   != rocfft_status_success)
                    throw rocfft_failure("rocfft failed to set/update a work buffer");
            }
            // i/o data sizes
            in_bytes = sizeof(hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::IN>)
                       * compute_ptrdiff(lengths_cm, istrides_cm, batch[0], idist[0]);
            out_bytes = sizeof(hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUT>)
                        * compute_ptrdiff(lengths_cm, ostrides_cm, batch[0], odist[0]);
            // set input & outut pointers and possible devices buffers
            update_io_ptr_and_buffers(user_in, user_out);
            return;
        }
    };

    template <rocfft_transform_type dft_type,
              rocfft_precision      prec,
              size_t                rank,
              typename T,
              size_t batch_rank = 1>
    hipfftw_plan_internal*
        hipfftw_create_plan(const std::array<T, rank>&                                  lengths_rm,
                            const std::array<T, rank>&                                  istrides_rm,
                            const std::array<T, rank>&                                  ostrides_rm,
                            hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::IN>*  user_in,
                            hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUT>* user_out,
                            const std::array<T, batch_rank>&                            batch,
                            const std::array<T, batch_rank>&                            idist,
                            const std::array<T, batch_rank>&                            odist,
                            unsigned                                                    flags)
    {
        auto ret = std::make_unique<hipfftw_plan_internal>();
        ret->init<dft_type, prec, rank, T, batch_rank>(
            lengths_rm, istrides_rm, ostrides_rm, user_in, user_out, batch, idist, odist, flags);
        return ret.release();
    }

    template <size_t rank,
              size_t batch_rank,
              typename T,
              std::enable_if_t<std::is_integral_v<T> && (rank > 0 & batch_rank > 0), bool> = true>
    struct hipfftw_general_layout_data
    {
        std::array<T, rank>       lengths;
        std::array<T, rank>       istrides;
        std::array<T, rank>       ostrides;
        std::array<T, batch_rank> batches;
        std::array<T, batch_rank> idist;
        std::array<T, batch_rank> odist;
        // constexpr getters
        constexpr inline size_t get_rank() const
        {
            return rank;
        }
        constexpr inline size_t get_batch_rank() const
        {
            return batch_rank;
        }
    };

    template <size_t rank, rocfft_transform_type dft_type, typename T>
    hipfftw_general_layout_data<rank, 1, T> hipfftw_get_default_data_layout_info_rm(
        bool is_in_place, const std::array<T, rank>& user_lengths_rm, size_t batch_sz = 1)
    {
        auto overflow_guarded_mult = [](T& a, const T& b) {
            bool overflowing = false;
            if(a > 0 && b >= 0)
                overflowing = b > std::numeric_limits<T>::max() / a;
            else if(a < 0 && b <= 0)
                overflowing = b < std::numeric_limits<T>::max() / a;
            else if(a > 0)
                overflowing = b < std::numeric_limits<T>::min() / a;
            else if(a < 0)
                overflowing = b > std::numeric_limits<T>::min() / a;
            if(overflowing)
                throw hipfftw_invalid_arg(
                    "length(s) trigger overflows for stride and/or distance calculations");
            a *= b;
        };

        hipfftw_general_layout_data<rank, 1, T> ret;
        ret.lengths = user_lengths_rm;
        T ival = 1, oval = 1;
        for(int dim_idx = static_cast<int>(rank) - 1; dim_idx >= 0; dim_idx--)
        {
            ret.istrides[dim_idx] = ival;
            ret.ostrides[dim_idx] = oval;
            if(is_real(dft_type) && dim_idx == rank - 1)
            {
                T& cmplx_stride_val = dft_type == rocfft_transform_type_real_forward ? oval : ival;
                T& real_stride_val  = dft_type == rocfft_transform_type_real_forward ? ival : oval;
                overflow_guarded_mult(cmplx_stride_val, ret.lengths[dim_idx] / 2 + 1);
                if(is_in_place)
                    overflow_guarded_mult(real_stride_val, 2 * (ret.lengths[dim_idx] / 2 + 1));
                else
                    overflow_guarded_mult(real_stride_val, ret.lengths[dim_idx]);
            }
            else
            {
                overflow_guarded_mult(ival, ret.lengths[dim_idx]);
                overflow_guarded_mult(oval, ret.lengths[dim_idx]);
            }
        }
        ret.batches[0] = batch_sz;
        ret.idist[0]   = ival;
        ret.odist[0]   = oval;
        return ret;
    }

    template <rocfft_transform_type dft_type, rocfft_precision prec, size_t rank, typename T>
    hipfftw_plan_internal* hipfftw_create_default_unbatched_plan(
        const std::array<T, rank>&                                  n,
        hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::IN>*  in,
        hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUT>* out,
        unsigned                                                    flags)
    {
        auto layout_data = hipfftw_get_default_data_layout_info_rm<rank, dft_type>(
            static_cast<void*>(in) == static_cast<void*>(out), n);
        return hipfftw_create_plan<dft_type, prec, rank, T>(layout_data.lengths,
                                                            layout_data.istrides,
                                                            layout_data.ostrides,
                                                            in,
                                                            out,
                                                            layout_data.batches,
                                                            layout_data.idist,
                                                            layout_data.odist,
                                                            flags);
    }

    template <rocfft_transform_type dft_type, rocfft_precision prec, typename T>
    hipfftw_plan_internal* hipfftw_create_default_unbatched_plan(
        int                                                         rank,
        const T*                                                    n,
        hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::IN>*  in,
        hipfftw_user_data_t<dft_type, prec, hipfftw_io_label::OUT>* out,
        unsigned                                                    flags)
    {
        if(rank <= 0)
            throw hipfftw_invalid_arg("invalid rank");
        if(!n)
            throw hipfftw_invalid_arg("nullptr lengths argument");
        // rank == 1, 2, 3, or unsupported
        switch(rank)
        {
        case 1:
            return hipfftw_create_default_unbatched_plan<dft_type, prec, 1>(
                std::array<T, 1>({n[0]}), in, out, flags);
        case 2:
            return hipfftw_create_default_unbatched_plan<dft_type, prec, 2>(
                std::array<T, 2>({n[0], n[1]}), in, out, flags);
        case 3:
            return hipfftw_create_default_unbatched_plan<dft_type, prec, 3>(
                std::array<T, 3>({n[0], n[1], n[2]}), in, out, flags);
        default:
            throw hipfftw_unsupported("transforms of rank > 3 are not supported");
        }
        // unreachable
    }

    inline void hipfftw_validate_sign(int sign)
    {
        if(sign != FFTW_FORWARD && sign != FFTW_BACKWARD)
            throw hipfftw_invalid_arg("invalid sign");
    }

    template <rocfft_precision prec, size_t rank, typename T>
    hipfftw_plan_internal*
        hipfftw_create_default_unbatched_complex_plan(const std::array<T, rank>&    n,
                                                      int                           sign,
                                                      hipfftw_complex_data_t<prec>* in,
                                                      hipfftw_complex_data_t<prec>* out,
                                                      unsigned                      flags)
    {
        hipfftw_validate_sign(sign);
        if(sign == FFTW_FORWARD)
            return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_forward,
                                                         prec,
                                                         rank,
                                                         T>(n, in, out, flags);
        else
            return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_inverse,
                                                         prec,
                                                         rank,
                                                         T>(n, in, out, flags);
    }

    template <rocfft_precision prec, typename T>
    hipfftw_plan_internal*
        hipfftw_create_default_unbatched_complex_plan(int                           rank,
                                                      const T*                      n,
                                                      int                           sign,
                                                      hipfftw_complex_data_t<prec>* in,
                                                      hipfftw_complex_data_t<prec>* out,
                                                      unsigned                      flags)
    {
        hipfftw_validate_sign(sign);
        if(sign == FFTW_FORWARD)
            return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_forward,
                                                         prec,
                                                         T>(rank, n, in, out, flags);
        else
            return hipfftw_create_default_unbatched_plan<rocfft_transform_type_complex_inverse,
                                                         prec,
                                                         T>(rank, n, in, out, flags);
    }

    bool hipfftw_handler_is_verbose() noexcept
    try
    {
        const char* env = std::getenv("HIPFFTW_LOG_EXCEPTIONS");
        return env && std::stoi(env) > 0;
    }
    catch(...)
    {
        return false;
    }

    inline void hipfftw_exception_handler() noexcept
    try
    {
        throw;
    }
    catch(const hipfftw_invalid_arg& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "Invalid argument detected; what(): " << e.what() << std::endl;
    }
    catch(const hipfftw_unsupported& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "Feature is not supported; what(): " << e.what() << std::endl;
    }
    catch(const hipfftw_invalid_plan& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "An invalid plan was used; what(): " << e.what() << std::endl;
    }
    catch(const hipfftw_internal_error& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "An error happened internally to hipfftw; what(): " << e.what()
                      << std::endl;
    }
    catch(const rocfft_failure& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "A rocfft failure happened; what(): " << e.what() << std::endl;
    }
    catch(const hipfftw_bad_gpu_alloc& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "GPU allocation failure caught; what(): " << e.what()
                      << "\nThe attempted size was " << e.attempted_size << " bytes" << std::endl;
    }
    catch(const hipfftw_bad_alloc& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "Allocation failure caught; what(): " << e.what()
                      << "\nThe attempted size was " << e.attempted_size << " bytes" << std::endl;
    }
    catch(const hipfftw_execution_error& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "Execution error detected; what(): " << e.what() << std::endl;
    }
    catch(const hipfftw_flow_redirection& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "Redirecting execution flow; what(): " << e.what() << std::endl;
    }
    catch(const std::runtime_error& e)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "Runtime error detected; what(): " << e.what() << std::endl;
    }
    catch(...)
    {
        if(hipfftw_handler_is_verbose())
            std::cerr << "An unidentified exception was caught..." << std::endl;
    }

    constexpr size_t no_limit = std::numeric_limits<size_t>::max();
    template <hipMemoryType type>
    size_t hipfftw_alloc_host_limit() noexcept
    try
    {
        static_assert(type == hipMemoryType::hipMemoryTypeHost
                      || type == hipMemoryType::hipMemoryTypeUnregistered);
        const char* env = nullptr;
        if constexpr(type == hipMemoryType::hipMemoryTypeHost)
            env = std::getenv("HIPFFTW_ALLOC_LIMIT_PINNED_HOST");
        else
            env = std::getenv("HIPFFTW_ALLOC_LIMIT_PAGEABLE_HOST");
        if(!env)
            throw int(); // no limit set
        const unsigned long long desired_limit = std::stoull(env);
        return desired_limit > no_limit ? no_limit : static_cast<size_t>(desired_limit);
    }
    catch(...)
    {
        // no actual limit set, or failed to read it
        return no_limit;
    }

    // possible TODO for hipfftw_alloc_host_accessible: consider using hipMallocManaged,
    // first, if the device supports it [LINUX ONLY].
    // NOTE: a limit may be set for any attempted kind of allocation via a dedicated
    // environment variable. If the request exceeds that limit or if the attempted allocation
    // otherwise fails, a lesser-ranked kind of allocation is attempted instead.
    template <typename element_type = void, hipMemoryType type = hipMemoryType::hipMemoryTypeHost>
    element_type* hipfftw_alloc_host_accessible(size_t num_elements) noexcept
    {
        static_assert(type == hipMemoryType::hipMemoryTypeHost
                      || type == hipMemoryType::hipMemoryTypeUnregistered);
        void* ret = nullptr;
        try
        {
            const size_t byte_size
                = num_elements
                  * sizeof(
                      std::conditional_t<std::is_same_v<void, element_type>, char, element_type>);
            if(byte_size > 0)
            {
                if(byte_size > hipfftw_alloc_host_limit<type>())
                {
                    throw hipfftw_flow_redirection("requested size exceeds set limit. Forcing a "
                                                   "lesser-ranked allocation type.");
                }
                if constexpr(type == hipMemoryType::hipMemoryTypeHost)
                {
                    if(hipHostMalloc(&ret, byte_size) != hipSuccess || !ret)
                    {
                        throw hipfftw_bad_alloc("creation of pinned host allocation failed",
                                                byte_size);
                    }
                }
                else
                {
                    // using a conservative alignemnt
                    constexpr size_t alignment = 1024;

                    ret = std::aligned_alloc(alignment, byte_size);
                    if(!ret)
                        throw hipfftw_bad_alloc("creation of host allocation failed", byte_size);
                }
            }
        }
        catch(...)
        {
            hipfftw_exception_handler();
            if constexpr(type == hipMemoryType::hipMemoryTypeHost)
            {
                // pinned host allocation failed or was not approved
                // --> attempt pageable allocation
                return hipfftw_alloc_host_accessible<element_type,
                                                     hipMemoryType::hipMemoryTypeUnregistered>(
                    num_elements);
            }
            else
            {
                // no other fallback
                ret = nullptr;
            }
        }
        return static_cast<element_type*>(ret);
    }

    void hipfftw_free(void* ptr) noexcept
    {
        try
        {
            // create an owning data pointer bundle on given pointed to leverage
            // the structure's destructor as it goes out of scope
            hipfftw_data_ptr_bundle<hipfftw_owns_it> bundle(ptr);
        }
        catch(...)
        {
            hipfftw_exception_handler();
        }
    }
} // end of implementation details

void* fftw_malloc(size_t n)
{
    return hipfftw_alloc_host_accessible(n);
}

void* fftwf_malloc(size_t n)
{
    return hipfftw_alloc_host_accessible(n);
}

double* fftw_alloc_real(size_t n)
{
    return hipfftw_alloc_host_accessible<double>(n);
}

fftw_complex* fftw_alloc_complex(size_t n)
{
    return hipfftw_alloc_host_accessible<fftw_complex>(n);
}

float* fftwf_alloc_real(size_t n)
{
    return hipfftw_alloc_host_accessible<float>(n);
}

fftwf_complex* fftwf_alloc_complex(size_t n)
{
    return hipfftw_alloc_host_accessible<fftwf_complex>(n);
}

void fftw_free(void* p)
{
    hipfftw_free(p);
}

void fftwf_free(void* p)
{
    hipfftw_free(p);
}

void fftw_destroy_plan(fftw_plan plan)
{
    auto internal_plan = static_cast<hipfftw_plan_internal*>(plan);
    delete internal_plan;
}

void fftwf_destroy_plan(fftwf_plan plan)
{
    auto internal_plan = static_cast<hipfftw_plan_internal*>(plan);
    delete internal_plan;
}

void fftw_cleanup() {}

void fftwf_cleanup() {}

void fftw_execute(const fftwf_plan plan)
try
{
    auto internal_plan = static_cast<const hipfftw_plan_internal*>(plan);
    if(!internal_plan)
        throw hipfftw_invalid_plan("nullptr plan used at execution");
    internal_plan->execute();
}
catch(...)
{
    hipfftw_exception_handler();
    return;
}

void fftwf_execute(const fftwf_plan plan)
try
{
    auto internal_plan = static_cast<const hipfftw_plan_internal*>(plan);
    if(!internal_plan)
        throw hipfftw_invalid_plan("nullptr plan used at execution");
    internal_plan->execute();
}
catch(...)
{
    hipfftw_exception_handler();
    return;
}

fftw_plan fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 1;
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 1;
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan
    fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 2;
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_2d(
    int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 2;
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_3d(
    int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 3;
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1, n2}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_3d(
    int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr int  rank = 3;
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec, rank>(
        std::array<int, rank>({n0, n1, n2}), sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft(
    int rank, const int* n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags)
try
{
    constexpr auto prec = rocfft_precision_double;
    return hipfftw_create_default_unbatched_complex_plan<prec>(rank, n, sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft(
    int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags)
try
{
    constexpr auto prec = rocfft_precision_single;
    return hipfftw_create_default_unbatched_complex_plan<prec>(rank, n, sign, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c_1d(int n, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan
    fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_r2c(int rank, const int* n, float* in, fftwf_complex* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_forward;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r_1d(int n, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr int  rank     = 1;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr int  rank     = 2;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan
    fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan
    fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr int  rank     = 3;
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec, rank>(
        std::array<int, rank>({n0, n1, n2}), in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftw_plan fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_double;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
    return nullptr;
}

fftwf_plan fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, float* out, unsigned flags)
try
{
    constexpr auto dft_type = rocfft_transform_type_real_inverse;
    constexpr auto prec     = rocfft_precision_single;
    return hipfftw_create_default_unbatched_plan<dft_type, prec>(rank, n, in, out, flags);
}
catch(...)
{
    hipfftw_exception_handler();
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
