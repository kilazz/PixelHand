// =======================================================================================
//  decoder.cpp - High-Performance C++ Python Module for DDS File Decoding
// =======================================================================================

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <limits>
#include <future> // Added for std::async

#define NOMINMAX
#include <Windows.h>
#include <objbase.h>

#include "DirectXTex.h"
// Include needed for XMConvertHalfToFloatStream
#include <DirectXPackedVector.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace DirectX;

// =======================================================================================
// Global Constants and Type Definitions
// =======================================================================================

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3) \
    ((uint32_t)(uint8_t)(ch0) | ((uint32_t)(uint8_t)(ch1) << 8) | \
    ((uint32_t)(uint8_t)(ch2) << 16) | ((uint32_t)(uint8_t)(ch3) << 24))
#endif

// DDS pixel format flags
constexpr uint32_t DDPF_ALPHAPIXELS = 0x00000001;
constexpr uint32_t DDPF_ALPHA       = 0x00000002;
constexpr uint32_t DDPF_FOURCC      = 0x00000004;
constexpr uint32_t DDPF_RGB         = 0x00000040;
constexpr uint32_t DDPF_LUMINANCE   = 0x00020000;

// DDS header flags
constexpr uint32_t DDSD_MIPMAPCOUNT = 0x00020000;
constexpr uint32_t DDSD_DEPTH       = 0x00800000;

// DDS caps2 flags
constexpr uint32_t DDSCAPS2_CUBEMAP = 0x00000200;
constexpr uint32_t DDSCAPS2_VOLUME  = 0x00200000;

// Standard FourCC codes
constexpr uint32_t FOURCC_DXT1 = MAKEFOURCC('D', 'X', 'T', '1');
constexpr uint32_t FOURCC_DXT2 = MAKEFOURCC('D', 'X', 'T', '2');
constexpr uint32_t FOURCC_DXT3 = MAKEFOURCC('D', 'X', 'T', '3');
constexpr uint32_t FOURCC_DXT4 = MAKEFOURCC('D', 'X', 'T', '4');
constexpr uint32_t FOURCC_DXT5 = MAKEFOURCC('D', 'X', 'T', '5');
constexpr uint32_t FOURCC_DX10 = MAKEFOURCC('D', 'X', '1', '0');
constexpr uint32_t FOURCC_ATI1 = MAKEFOURCC('A', 'T', 'I', '1');
constexpr uint32_t FOURCC_BC4U = MAKEFOURCC('B', 'C', '4', 'U');
constexpr uint32_t FOURCC_BC4S = MAKEFOURCC('B', 'C', '4', 'S');
constexpr uint32_t FOURCC_ATI2 = MAKEFOURCC('A', 'T', 'I', '2');
constexpr uint32_t FOURCC_BC5U = MAKEFOURCC('B', 'C', '5', 'U');
constexpr uint32_t FOURCC_BC5S = MAKEFOURCC('B', 'C', '5', 'S');

// CryEngine markers
constexpr uint32_t FOURCC_CRYF = MAKEFOURCC('C', 'R', 'Y', 'F');
constexpr uint32_t FOURCC_FYRC = MAKEFOURCC('F', 'Y', 'R', 'C');

#pragma pack(push, 1)
struct DDS_PIXELFORMAT {
    uint32_t dwSize, dwFlags, dwFourCC, dwRGBBitCount, dwRBitMask, dwGBitMask, dwBBitMask, dwABitMask;
};
struct DDS_HEADER {
    uint32_t dwSize, dwFlags, dwHeight, dwWidth, dwPitchOrLinearSize, dwDepth, dwMipMapCount;
    uint32_t dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t dwCaps, dwCaps2, dwCaps3, dwCaps4, dwReserved2;
};
struct DDS_HEADER_DXT10 {
    DXGI_FORMAT dxgiFormat;
    uint32_t resourceDimension, miscFlag, arraySize, miscFlags2;
};
#pragma pack(pop)

// Forward declarations
namespace {
std::string DXGIFormatToString(DXGI_FORMAT format);
std::string HResultToString(HRESULT hr);


// =======================================================================================
// Error Handling & COM Initialization
// =======================================================================================

std::string HResultToString(HRESULT hr) {
    char* msg_buf = nullptr;
    DWORD result = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&msg_buf), 0, nullptr);
    std::string msg;
    if (result > 0 && msg_buf) {
        msg = msg_buf;
        LocalFree(msg_buf);
        while (!msg.empty() && (msg.back() == '\r' || msg.back() == '\n' || msg.back() == ' ')) msg.pop_back();
    } else {
        std::ostringstream oss;
        oss << "HRESULT 0x" << std::hex << std::uppercase << hr;
        msg = oss.str();
    }
    return msg;
}

class DDSLoadError : public std::runtime_error {
public:
    DDSLoadError(const std::string& message, HRESULT hr = S_OK)
        : std::runtime_error(message + (FAILED(hr) ? " (" + HResultToString(hr) + ")" : "")), hresult_(hr) {}
    HRESULT get_hresult() const { return hresult_; }
private:
    HRESULT hresult_;
};

inline void ThrowIfFailed(HRESULT hr, const std::string& message) {
    if (FAILED(hr)) {
        throw DDSLoadError(message, hr);
    }
}

class CoInitializer {
public:
    CoInitializer() {
        hr_ = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (hr_ == RPC_E_CHANGED_MODE) hr_ = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        should_uninit_ = SUCCEEDED(hr_);
        if (hr_ == S_FALSE) { hr_ = S_OK; should_uninit_ = false; }
    }
    ~CoInitializer() { if (should_uninit_) CoUninitialize(); }
    bool IsValid() const { return SUCCEEDED(hr_); }
    HRESULT GetResult() const { return hr_; }
private:
    HRESULT hr_;
    bool should_uninit_ = true;
};

// =======================================================================================
//  Legacy Format Utilities Namespace (Xbox, CryEngine)
// =======================================================================================
namespace LegacyUtils {
    template<typename T>
    T SwapBytes(T value) {
        static_assert(std::is_integral<T>::value, "SwapBytes can only be used with integral types");
        T result{};
        auto* src = reinterpret_cast<uint8_t*>(&value);
        auto* dst = reinterpret_cast<uint8_t*>(&result);
        for (size_t i = 0; i < sizeof(T); ++i) {
            dst[i] = src[sizeof(T) - 1 - i];
        }
        return result;
    }

    static void EndianSwapBuffer(uint8_t* data, size_t dataSize, size_t elementSize) {
        if (elementSize <= 1) return;
        for (size_t i = 0; i + elementSize <= dataSize; i += elementSize) {
            for (size_t j = 0; j < elementSize / 2; ++j) {
                std::swap(data[i + j], data[i + elementSize - 1 - j]);
            }
        }
    }

    static bool IsXbox360Format(uint32_t fourCC) {
        switch (fourCC) {
            case MAKEFOURCC('1','T','X','D'):
            case MAKEFOURCC('3','T','X','D'):
            case MAKEFOURCC('5','T','X','D'):
            case MAKEFOURCC('0','1','X','D'):
            case MAKEFOURCC('1','I','T','A'):
            case MAKEFOURCC('2','I','T','A'):
            case MAKEFOURCC('U','4','C','B'):
            case MAKEFOURCC('U','5','C','B'):
                return true;
            default: return false;
        }
    }

    static uint32_t Xbox360ToStandardFourCC(uint32_t fourCC) {
        switch (fourCC) {
            case MAKEFOURCC('1','T','X','D'): return MAKEFOURCC('D','X','T','1');
            case MAKEFOURCC('3','T','X','D'): return MAKEFOURCC('D','X','T','3');
            case MAKEFOURCC('5','T','X','D'): return MAKEFOURCC('D','X','T','5');
            case MAKEFOURCC('0','1','X','D'): return MAKEFOURCC('D','X','1','0');
            case MAKEFOURCC('1','I','T','A'): return MAKEFOURCC('A','T','I','1');
            case MAKEFOURCC('2','I','T','A'): return MAKEFOURCC('A','T','I','2');
            case MAKEFOURCC('U','4','C','B'): return MAKEFOURCC('B','C','4','U');
            case MAKEFOURCC('U','5','C','B'): return MAKEFOURCC('B','C','5','U');
            default: return fourCC;
        }
    }

    static void PerformXboxEndianSwap(uint8_t* data, size_t dataSize, DXGI_FORMAT format) {
        if (!data || dataSize == 0) return;
        if (DirectX::IsCompressed(format)) {
            size_t blockSize = (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC4_UNORM) ? 8 : 16;
            for (size_t i = 0; i + blockSize <= dataSize; i += blockSize) {
                auto* block16 = reinterpret_cast<uint16_t*>(data + i);
                if (format == DXGI_FORMAT_BC1_UNORM || format == DXGI_FORMAT_BC4_UNORM) {
                    block16[0] = SwapBytes(block16[0]); block16[1] = SwapBytes(block16[1]);
                } else if (format == DXGI_FORMAT_BC2_UNORM || format == DXGI_FORMAT_BC3_UNORM) {
                    block16[4] = SwapBytes(block16[4]); block16[5] = SwapBytes(block16[5]);
                    if (format == DXGI_FORMAT_BC3_UNORM) {
                        block16[1] = SwapBytes(block16[1]); block16[2] = SwapBytes(block16[2]); block16[3] = SwapBytes(block16[3]);
                    }
                } else if (format == DXGI_FORMAT_BC5_UNORM) {
                    block16[1] = SwapBytes(block16[1]); block16[2] = SwapBytes(block16[2]); block16[3] = SwapBytes(block16[3]);
                    block16[5] = SwapBytes(block16[5]); block16[6] = SwapBytes(block16[6]); block16[7] = SwapBytes(block16[7]);
                }
            }
        } else {
            size_t bpp = DirectX::BitsPerPixel(format);
            if (bpp >= 16) {
                EndianSwapBuffer(data, dataSize, bpp / 8);
            }
        }
    }

    static void UnswizzleBlockLinear(const uint8_t* src, uint8_t* dst, uint32_t width, uint32_t height, uint32_t blockBytes, size_t srcSize, size_t dstSize) {
        uint32_t blockWidth = (width + 3) / 4;
        uint32_t blockHeight = (height + 3) / 4;
        size_t requiredSize = static_cast<size_t>(blockWidth) * blockHeight * blockBytes;
        if (srcSize < requiredSize || dstSize < requiredSize) throw DDSLoadError("Buffer too small for unswizzle operation.");

        uint32_t logW = (blockWidth > 1) ? static_cast<uint32_t>(floor(log2(blockWidth - 1))) + 1 : 0;
        uint32_t logH = (blockHeight > 1) ? static_cast<uint32_t>(floor(log2(blockHeight - 1))) + 1 : 0;
        uint32_t min_log = std::min(logW, logH);

        for (uint32_t y = 0; y < blockHeight; ++y) {
            for (uint32_t x = 0; x < blockWidth; ++x) {
                uint32_t swizzledIndex = 0;
                for (uint32_t i = 0; i < min_log; ++i) {
                    swizzledIndex |= ((x >> i) & 1) << (2 * i);
                    swizzledIndex |= ((y >> i) & 1) << (2 * i + 1);
                }
                if (logW > logH) {
                    for (uint32_t i = min_log; i < logW; ++i) swizzledIndex |= ((x >> i) & 1) << (i + min_log);
                } else {
                    for (uint32_t i = min_log; i < logH; ++i) swizzledIndex |= ((y >> i) & 1) << (i + min_log);
                }
                size_t srcOffset = static_cast<size_t>(swizzledIndex) * blockBytes;
                size_t dstOffset = (static_cast<size_t>(y) * blockWidth + x) * blockBytes;
                if (srcOffset + blockBytes <= srcSize && dstOffset + blockBytes <= dstSize) {
                    memcpy(dst + dstOffset, src + srcOffset, blockBytes);
                }
            }
        }
    }
} // namespace LegacyUtils

// =======================================================================================
// NumPy Conversion Utilities Namespace
// =======================================================================================
namespace NumpyUtils {
    static float half_to_float(uint16_t half) {
        return DirectX::PackedVector::XMConvertHalfToFloat(half);
    }

    template <typename T>
    nb::object direct_copy_to_numpy(const std::vector<std::unique_ptr<ScratchImage>>& images, int channels) {
        size_t num_images = images.size();
        const auto* first = images[0]->GetImage(0, 0, 0);
        size_t ndim = 0;
        size_t shape[4] = {0};
        if (num_images > 1) shape[ndim++] = num_images;
        shape[ndim++] = first->height;
        shape[ndim++] = first->width;
        if (channels > 0) shape[ndim++] = channels;

        size_t elements_per_slice = first->height * first->width * (channels > 0 ? channels : 1);
        T* ptr = new T[num_images * elements_per_slice];
        nb::capsule owner(ptr, [](void *p) noexcept { delete[] (T *) p; });
        nb::ndarray<nb::numpy, T> arr(ptr, ndim, shape, owner);

        size_t numpy_row_pitch = first->width * sizeof(T) * (channels > 0 ? channels : 1);

        for (size_t i = 0; i < num_images; ++i) {
            const auto* img = images[i]->GetImage(0, 0, 0);
            T* dst_slice = ptr + i * elements_per_slice;
            if (img->rowPitch == numpy_row_pitch) {
                memcpy(dst_slice, img->pixels, img->slicePitch);
            } else {
                auto* dst_ptr = reinterpret_cast<uint8_t*>(dst_slice);
                const auto* src_ptr = img->pixels;
                for (size_t y = 0; y < img->height; ++y) {
                    memcpy(dst_ptr + y * numpy_row_pitch, src_ptr + y * img->rowPitch, numpy_row_pitch);
                }
            }
        }
        return nb::cast(arr);
    }

    static nb::object convert_half_to_float_numpy(const std::vector<std::unique_ptr<ScratchImage>>& images, size_t num_components) {
        size_t num_images = images.size();
        const auto* first = images[0]->GetImage(0, 0, 0);
        size_t ndim = 0;
        size_t shape[4] = {0};
        if (num_images > 1) shape[ndim++] = num_images;
        shape[ndim++] = first->height;
        shape[ndim++] = first->width;
        if (num_components > 1) shape[ndim++] = num_components;

        size_t elements_per_slice = first->height * first->width * num_components;
        float* ptr = new float[num_images * elements_per_slice];
        nb::capsule owner(ptr, [](void *p) noexcept { delete[] (float *) p; });
        nb::ndarray<nb::numpy, float> arr(ptr, ndim, shape, owner);

        for (size_t i = 0; i < num_images; ++i) {
            const auto* img = images[i]->GetImage(0, 0, 0);
            float* dst_slice = ptr + i * elements_per_slice;
            const auto* src_ptr_base = img->pixels;
            for (size_t y = 0; y < img->height; ++y) {
                const auto* src_row_ptr = reinterpret_cast<const uint16_t*>(src_ptr_base + y * img->rowPitch);
                float* dst_row_ptr = dst_slice + y * img->width * num_components;
                DirectX::PackedVector::XMConvertHalfToFloatStream(
                    dst_row_ptr, sizeof(float),
                    reinterpret_cast<const DirectX::PackedVector::HALF*>(src_row_ptr), sizeof(uint16_t),
                    img->width * num_components
                );
            }
        }
        return nb::cast(arr);
    }

    template <typename T, typename Func>
    nb::object custom_convert_to_numpy(const std::vector<std::unique_ptr<ScratchImage>>& images, int channels, Func convert_row) {
        size_t num_images = images.size();
        const auto* first = images[0]->GetImage(0, 0, 0);
        size_t ndim = 0;
        size_t shape[4] = {0};
        if (num_images > 1) shape[ndim++] = num_images;
        shape[ndim++] = first->height;
        shape[ndim++] = first->width;
        if (channels > 0) shape[ndim++] = channels;

        size_t elements_per_slice = first->height * first->width * (channels > 0 ? channels : 1);
        T* ptr = new T[num_images * elements_per_slice];
        nb::capsule owner(ptr, [](void *p) noexcept { delete[] (T *) p; });
        nb::ndarray<nb::numpy, T> arr(ptr, ndim, shape, owner);

        for (size_t i = 0; i < num_images; ++i) {
            const auto* img = images[i]->GetImage(0, 0, 0);
            T* dst_slice = ptr + i * elements_per_slice;
            for (size_t y = 0; y < img->height; ++y) {
                const uint8_t* src_row = img->pixels + y * img->rowPitch;
                T* dst_row = dst_slice + y * img->width * channels;
                convert_row(src_row, dst_row, img->width, img->format);
            }
        }
        return nb::cast(arr);
    }

    static nb::object CreateNumpyArrayFromImages(const std::vector<std::unique_ptr<ScratchImage>>& images) {
        if (images.empty()) throw DDSLoadError("No images to convert.");
        const auto* first = images[0]->GetImage(0, 0, 0);
        if (!first || !first->pixels) throw DDSLoadError("Invalid image data.");

        switch (first->format) {
            case DXGI_FORMAT_R32G32B32A32_FLOAT: return direct_copy_to_numpy<float>(images, 4);
            case DXGI_FORMAT_R32G32B32_FLOAT:    return direct_copy_to_numpy<float>(images, 3);
            case DXGI_FORMAT_R32G32_FLOAT:       return direct_copy_to_numpy<float>(images, 2);
            case DXGI_FORMAT_R32_FLOAT:          return direct_copy_to_numpy<float>(images, 1);
            case DXGI_FORMAT_R16G16B16A16_FLOAT: return convert_half_to_float_numpy(images, 4);
            case DXGI_FORMAT_R16G16_FLOAT:       return convert_half_to_float_numpy(images, 2);
            case DXGI_FORMAT_R16_FLOAT:          return convert_half_to_float_numpy(images, 1);
            case DXGI_FORMAT_R16G16B16A16_UNORM: return direct_copy_to_numpy<uint16_t>(images, 4);
            case DXGI_FORMAT_R16G16_UNORM:       return direct_copy_to_numpy<uint16_t>(images, 2);
            case DXGI_FORMAT_R16_UNORM:          return direct_copy_to_numpy<uint16_t>(images, 1);
            case DXGI_FORMAT_R16G16B16A16_UINT:  return direct_copy_to_numpy<uint16_t>(images, 4);
            case DXGI_FORMAT_R16G16_UINT:        return direct_copy_to_numpy<uint16_t>(images, 2);
            case DXGI_FORMAT_R16_UINT:           return direct_copy_to_numpy<uint16_t>(images, 1);
            case DXGI_FORMAT_R16G16B16A16_SINT:  return direct_copy_to_numpy<int16_t>(images, 4);
            case DXGI_FORMAT_R16G16_SINT:        return direct_copy_to_numpy<int16_t>(images, 2);
            case DXGI_FORMAT_R16_SINT:           return direct_copy_to_numpy<int16_t>(images, 1);
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
                return direct_copy_to_numpy<uint8_t>(images, 4);
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8X8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    for (size_t x = 0; x < w; ++x) {
                        dst[x * 4 + 0] = src[x * 4 + 2];
                        dst[x * 4 + 1] = src[x * 4 + 1];
                        dst[x * 4 + 2] = src[x * 4 + 0];
                        dst[x * 4 + 3] = src[x * 4 + 3];
                    }
                });
            case DXGI_FORMAT_R8G8B8A8_UINT: return direct_copy_to_numpy<uint8_t>(images, 4);
            case DXGI_FORMAT_R8G8_UINT:     return direct_copy_to_numpy<uint8_t>(images, 2);
            case DXGI_FORMAT_R8_UINT:       return direct_copy_to_numpy<uint8_t>(images, 1);
            case DXGI_FORMAT_R8G8B8A8_SINT: return direct_copy_to_numpy<int8_t>(images, 4);
            case DXGI_FORMAT_R8G8_SINT:     return direct_copy_to_numpy<int8_t>(images, 2);
            case DXGI_FORMAT_R8_SINT:       return direct_copy_to_numpy<int8_t>(images, 1);
            case DXGI_FORMAT_R8G8B8A8_SNORM:
            case DXGI_FORMAT_R8G8_SNORM:
            case DXGI_FORMAT_R8_SNORM:
            case DXGI_FORMAT_R16G16B16A16_SNORM:
            case DXGI_FORMAT_R16G16_SNORM:
            case DXGI_FORMAT_R16_SNORM: {
                int channels = (first->format == DXGI_FORMAT_R8G8B8A8_SNORM || first->format == DXGI_FORMAT_R16G16B16A16_SNORM) ? 4 : ((first->format == DXGI_FORMAT_R8G8_SNORM || first->format == DXGI_FORMAT_R16G16_SNORM) ? 2 : 1);
                return custom_convert_to_numpy<float>(images, channels, [channels](const uint8_t* src, float* dst, size_t w, DXGI_FORMAT fmt) {
                    bool is16bit = (fmt == DXGI_FORMAT_R16G16B16A16_SNORM || fmt == DXGI_FORMAT_R16G16_SNORM || fmt == DXGI_FORMAT_R16_SNORM);
                    for (size_t x = 0; x < w * channels; ++x) {
                        dst[x] = std::max(-1.f, (is16bit ? static_cast<float>(reinterpret_cast<const int16_t*>(src)[x]) / 32767.f : static_cast<float>(reinterpret_cast<const int8_t*>(src)[x]) / 127.f));
                    }
                });
            }
            case DXGI_FORMAT_B5G6R5_UNORM:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    const auto* src16 = reinterpret_cast<const uint16_t*>(src);
                    for (size_t x = 0; x < w; ++x) {
                        uint16_t p = src16[x];
                        dst[x * 4 + 0] = static_cast<uint8_t>(((p >> 11) & 0x1F) * 255/31);
                        dst[x * 4 + 1] = static_cast<uint8_t>(((p >> 5)  & 0x3F) * 255/63);
                        dst[x * 4 + 2] = static_cast<uint8_t>(( p        & 0x1F) * 255/31);
                        dst[x * 4 + 3] = 255;
                    }
                });
            case DXGI_FORMAT_B5G5R5A1_UNORM:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    const auto* src16 = reinterpret_cast<const uint16_t*>(src);
                    for (size_t x = 0; x < w; ++x) {
                        uint16_t p = src16[x];
                        dst[x * 4 + 0] = static_cast<uint8_t>(((p>>10) & 0x1F) * 255/31);
                        dst[x * 4 + 1] = static_cast<uint8_t>(((p>>5)  & 0x1F) * 255/31);
                        dst[x * 4 + 2] = static_cast<uint8_t>((p&0x1F) * 255/31);
                        dst[x * 4 + 3] = static_cast<uint8_t>(((p>>15) & 0x01) * 255);
                    }
                });
            case DXGI_FORMAT_B4G4R4A4_UNORM:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    const auto* src16 = reinterpret_cast<const uint16_t*>(src);
                    for (size_t x = 0; x < w; ++x) {
                        uint16_t p = src16[x];
                        uint8_t r = (p>>8) & 0x0F, g = (p>>4) & 0x0F, b=p&0x0F, a = (p>>12) & 0x0F;
                        dst[x * 4 + 0] = (r << 4) | r;
                        dst[x * 4 + 1] = (g << 4) | g;
                        dst[x * 4 + 2] = (b << 4) | b;
                        dst[x * 4 + 3] = (a << 4) | a;
                    }
                });
            case DXGI_FORMAT_R8G8_UNORM:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    for (size_t x = 0; x < w; ++x) {
                        dst[x * 4 + 0] = src[x * 2 + 0];
                        dst[x * 4 + 1] = src[x * 2 + 1];
                        dst[x * 4 + 2] = 0;
                        dst[x * 4 + 3] = 255;
                    }
                });
            case DXGI_FORMAT_R8_UNORM:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    for (size_t x = 0; x < w; ++x) {
                        uint8_t val = src[x];
                        dst[x * 4 + 0] = val;
                        dst[x * 4 + 1] = val;
                        dst[x * 4 + 2] = val;
                        dst[x * 4 + 3] = 255;
                    }
                });
            case DXGI_FORMAT_A8_UNORM:
                return custom_convert_to_numpy<uint8_t>(images, 4, [](const uint8_t* src, uint8_t* dst, size_t w, DXGI_FORMAT) {
                    for (size_t x = 0; x < w; ++x) {
                        dst[x * 4 + 0] = 255;
                        dst[x * 4 + 1] = 255;
                        dst[x * 4 + 2] = 255;
                        dst[x * 4 + 3] = src[x];
                    }
                });
            default:
                throw DDSLoadError("Unsupported DXGI_FORMAT for NumPy conversion: " + DXGIFormatToString(first->format));
        }
    }
} // namespace NumpyUtils


// =======================================================================================
// Core Loading and Decoding Logic
// =======================================================================================

struct AnalyzedHeaderInfo {
    DDS_HEADER header{};
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    const uint8_t* data_ptr = nullptr;
    size_t data_size = 0;
    bool is_swizzled = false;
    bool is_xbox = false;
};

AnalyzedHeaderInfo analyze_header(const uint8_t* dds_data_ptr, size_t dds_data_size) {
    if (dds_data_size < sizeof(DDS_HEADER) + 4) throw DDSLoadError("Input data too small for a DDS file");
    if (memcmp(dds_data_ptr, "DDS ", 4) != 0) throw DDSLoadError("Not a DDS file (missing 'DDS ' magic number)");

    AnalyzedHeaderInfo info;
    info.header = *reinterpret_cast<const DDS_HEADER*>(dds_data_ptr + 4);
    const uint8_t* file_end = dds_data_ptr + dds_data_size;
    if (info.header.dwSize != sizeof(DDS_HEADER)) throw DDSLoadError("Invalid DDS header size.");

    info.is_xbox = LegacyUtils::IsXbox360Format(info.header.ddspf.dwFourCC);

    const DDS_PIXELFORMAT& pf = info.header.ddspf;
    if (pf.dwSize != sizeof(DDS_PIXELFORMAT)) throw DDSLoadError("Invalid DDS_PIXELFORMAT size.");

    if (pf.dwFlags & DDPF_FOURCC) {
        uint32_t fourCC = info.is_xbox ? LegacyUtils::Xbox360ToStandardFourCC(pf.dwFourCC) : pf.dwFourCC;
        switch (fourCC) {
            case FOURCC_DXT1: info.format = DXGI_FORMAT_BC1_UNORM; break;
            case FOURCC_DXT2: case FOURCC_DXT3: info.format = DXGI_FORMAT_BC2_UNORM; break;
            case FOURCC_DXT4: case FOURCC_DXT5: info.format = DXGI_FORMAT_BC3_UNORM; break;
            case FOURCC_ATI1: case FOURCC_BC4U: info.format = DXGI_FORMAT_BC4_UNORM; break;
            case FOURCC_BC4S: info.format = DXGI_FORMAT_BC4_SNORM; break;
            case FOURCC_ATI2: case FOURCC_BC5U: info.format = DXGI_FORMAT_BC5_UNORM; break;
            case FOURCC_BC5S: info.format = DXGI_FORMAT_BC5_SNORM; break;
            case FOURCC_DX10: {
                const auto* ext = reinterpret_cast<const DDS_HEADER_DXT10*>(&info.header + 1);
                if (reinterpret_cast<const uint8_t*>(ext) + sizeof(DDS_HEADER_DXT10) > file_end) throw DDSLoadError("DX10 header is out of bounds.");
                info.format = ext->dxgiFormat;
                break;
            }
        }
    }
    else if (pf.dwFlags & DDPF_RGB) {
        switch (pf.dwRGBBitCount) {
            case 32:
                if (pf.dwRBitMask == 0x00ff0000 && pf.dwGBitMask == 0x0000ff00 && pf.dwBBitMask == 0x000000ff) info.format = DXGI_FORMAT_R8G8B8A8_UNORM;
                else if (pf.dwRBitMask == 0x000000ff && pf.dwGBitMask == 0x0000ff00 && pf.dwBBitMask == 0x00ff0000) info.format = DXGI_FORMAT_B8G8R8A8_UNORM;
                break;
            case 16:
                if (pf.dwRBitMask == 0xf800 && pf.dwGBitMask == 0x07e0 && pf.dwBBitMask == 0x001f) info.format = DXGI_FORMAT_B5G6R5_UNORM;
                if (pf.dwRBitMask == 0x7c00 && pf.dwGBitMask == 0x03e0 && pf.dwBBitMask == 0x001f) info.format = DXGI_FORMAT_B5G5R5A1_UNORM;
                if (pf.dwRBitMask == 0x0f00 && pf.dwGBitMask == 0x00f0 && pf.dwBBitMask == 0x000f) info.format = DXGI_FORMAT_B4G4R4A4_UNORM;
                break;
        }
    }
    else if ((pf.dwFlags & DDPF_LUMINANCE) && pf.dwRGBBitCount == 8) info.format = DXGI_FORMAT_R8_UNORM;
    else if ((pf.dwFlags & DDPF_ALPHAPIXELS) && pf.dwRGBBitCount == 8) info.format = DXGI_FORMAT_A8_UNORM;

    info.data_ptr = dds_data_ptr + 4 + sizeof(DDS_HEADER);
    if (pf.dwFourCC == FOURCC_DX10 || (info.is_xbox && LegacyUtils::Xbox360ToStandardFourCC(pf.dwFourCC) == FOURCC_DX10)) {
        info.data_ptr += sizeof(DDS_HEADER_DXT10);
    }

    if (info.data_ptr + 4 <= file_end) {
        const uint32_t* marker = reinterpret_cast<const uint32_t*>(info.data_ptr);
        if (*marker == FOURCC_CRYF || *marker == FOURCC_FYRC) {
            info.is_swizzled = (*marker == FOURCC_CRYF);
            info.data_ptr += (*marker == FOURCC_CRYF) ? 8 : 4;
            if (pf.dwFlags & DDPF_RGB) {
                const size_t actual_data_size = file_end - info.data_ptr;
                size_t expected_size_dxt1 = static_cast<size_t>(std::max(1u, (info.header.dwWidth + 3) / 4)) * std::max(1u, (info.header.dwHeight + 3) / 4) * 8;
                if (actual_data_size >= expected_size_dxt1) {
                    info.format = DXGI_FORMAT_BC1_UNORM;
                }
            }
        }
    }
    if (info.data_ptr >= file_end) throw DDSLoadError("Image data offset is out of bounds.");
    info.data_size = file_end - info.data_ptr;
    return info;
}

std::unique_ptr<ScratchImage> load_with_fallback(const uint8_t* dds_data_ptr, size_t dds_data_size) {
    AnalyzedHeaderInfo info = analyze_header(dds_data_ptr, dds_data_size);

    size_t row_pitch, slice_pitch;
    ThrowIfFailed(ComputePitch(info.format, info.header.dwWidth, info.header.dwHeight, row_pitch, slice_pitch), "ComputePitch failed in fallback");
    if (info.data_size < slice_pitch) throw DDSLoadError("Insufficient pixel data in fallback.");

    std::vector<uint8_t> persistent_buffer(slice_pitch);
    uint8_t* mutable_pixel_data = persistent_buffer.data();

    if (info.is_swizzled) {
        size_t blockBytes = IsCompressed(info.format) ? ((info.format == DXGI_FORMAT_BC1_UNORM || info.format == DXGI_FORMAT_BC4_UNORM) ? 8 : 16) : ((4 * 4 * BitsPerPixel(info.format)) / 8);
        LegacyUtils::UnswizzleBlockLinear(info.data_ptr, mutable_pixel_data, info.header.dwWidth, info.header.dwHeight, (uint32_t)blockBytes, info.data_size, slice_pitch);
    } else {
        memcpy(mutable_pixel_data, info.data_ptr, slice_pitch);
    }

    if (info.is_xbox) {
        LegacyUtils::PerformXboxEndianSwap(mutable_pixel_data, slice_pitch, info.format);
    }

    auto image = std::make_unique<ScratchImage>();
    Image manual_image{ info.header.dwWidth, info.header.dwHeight, info.format, row_pitch, slice_pitch, mutable_pixel_data };
    ThrowIfFailed(image->InitializeFromImage(manual_image), "Fallback ScratchImage initialization failed");

    return image;
}

nb::dict decode_dds(const nb::bytes& dds_bytes, size_t mip_level = 0, int array_index = -1, bool force_rgba8 = false) {
    const auto* dds_data_ptr = reinterpret_cast<const uint8_t*>(dds_bytes.c_str());
    size_t dds_data_size = dds_bytes.size();

    nb::gil_scoped_release gil_release;
    CoInitializer com_init;
    if (!com_init.IsValid()) {
        throw DDSLoadError("COM initialization failed", com_init.GetResult());
    }

    auto image = std::make_unique<ScratchImage>();
    HRESULT hr = LoadFromDDSMemory(dds_data_ptr, dds_data_size, DDS_FLAGS_NONE, nullptr, *image);

    if (FAILED(hr)) {
        try {
            image = load_with_fallback(dds_data_ptr, dds_data_size);
        } catch (const std::exception& e) {
            throw DDSLoadError(std::string("DDS loading failed on all paths. Fallback error: ") + e.what());
        }
    }

    const auto& final_metadata = image->GetMetadata();
    if (mip_level >= final_metadata.mipLevels) {
        throw std::invalid_argument("Mipmap level is out of bounds");
    }

    auto process_single_image = [&](const DirectX::Image* input_image) -> std::unique_ptr<ScratchImage> {
        if (!input_image) throw DDSLoadError("Input image for processing is null");
        auto current_image = std::make_unique<ScratchImage>();
        ThrowIfFailed(current_image->InitializeFromImage(*input_image), "Failed to initialize from selected image");

        if (IsCompressed(current_image->GetMetadata().format)) {
            auto temp = std::make_unique<ScratchImage>();
            ThrowIfFailed(Decompress(*current_image->GetImage(0, 0, 0), DXGI_FORMAT_UNKNOWN, *temp), "Decompression failed");
            current_image = std::move(temp);
        }
        if (force_rgba8) {
            DXGI_FORMAT current_format = current_image->GetMetadata().format;
            if (current_format != DXGI_FORMAT_R8G8B8A8_UNORM && current_format != DXGI_FORMAT_B8G8R8A8_UNORM) {
                auto temp = std::make_unique<ScratchImage>();
                ThrowIfFailed(Convert(*current_image->GetImage(0, 0, 0), DXGI_FORMAT_R8G8B8A8_UNORM, TEX_FILTER_DEFAULT, TEX_THRESHOLD_DEFAULT, *temp), "Format conversion to RGBA8 failed");
                current_image = std::move(temp);
            }
        }
        return current_image;
    };

    bool load_all = array_index < 0;
    bool is_3d = final_metadata.dimension == TEX_DIMENSION_TEXTURE3D;
    size_t num_images_to_process = load_all ? (is_3d ? final_metadata.depth : final_metadata.arraySize) : 1;
    if (!load_all && static_cast<size_t>(array_index) >= final_metadata.arraySize) {
        throw std::invalid_argument("Array index is out of bounds");
    }

    std::vector<std::unique_ptr<ScratchImage>> processed_images(num_images_to_process);
    std::vector<std::future<void>> futures;

    // Use async tasks for parallel decompression/conversion of array slices or volume depth layers
    for (size_t i = 0; i < num_images_to_process; ++i) {
        futures.push_back(std::async(std::launch::async, [&](size_t idx) {
            size_t current_index = load_all ? idx : static_cast<size_t>(array_index);
            const DirectX::Image* selected_image = image->GetImage(mip_level, current_index, 0);
            if (!selected_image) {
                throw DDSLoadError("Failed to get image slice for index " + std::to_string(current_index));
            }
            processed_images[idx] = process_single_image(selected_image);
        }, i));
    }

    // Wait for all processing to complete and propagate exceptions
    for (auto& f : futures) {
        f.get();
    }

    const DirectX::Image* first_final_image = processed_images[0]->GetImage(0, 0, 0);
    if (!first_final_image || !first_final_image->pixels) {
        throw DDSLoadError("Could not retrieve final processed image pixels");
    }

    nb::gil_scoped_acquire gil_acquire;
    nb::object numpy_array = NumpyUtils::CreateNumpyArrayFromImages(processed_images);

    nb::dict result_dict;
    result_dict["width"] = final_metadata.width;
    result_dict["height"] = final_metadata.height;
    result_dict["depth"] = (is_3d ? final_metadata.depth : 1);
    result_dict["array_size"] = (!is_3d ? final_metadata.arraySize : 1);
    result_dict["data"] = numpy_array;
    result_dict["format_str"] = DXGIFormatToString(first_final_image->format);
    result_dict["mip_levels"] = final_metadata.mipLevels;
    result_dict["is_cubemap"] = final_metadata.IsCubemap();
    return result_dict;
}

nb::dict get_dds_metadata(const nb::bytes& dds_bytes) {
    const auto* dds_data_ptr = reinterpret_cast<const uint8_t*>(dds_bytes.c_str());
    size_t dds_data_size = dds_bytes.size();

    nb::gil_scoped_release gil_release;
    CoInitializer com_init;
    if (!com_init.IsValid()) {
        throw DDSLoadError("COM initialization failed", com_init.GetResult());
    }

    TexMetadata metadata;
    // Fix GetMetadataFromDDSMemory call ambiguity
    HRESULT hr = GetMetadataFromDDSMemory(dds_data_ptr, dds_data_size, DDS_FLAGS_NONE, metadata);

    if (FAILED(hr)) {
        try {
            AnalyzedHeaderInfo header_info = analyze_header(dds_data_ptr, dds_data_size);
            nb::gil_scoped_acquire gil_acquire;
            nb::dict d;
            d["width"] = header_info.header.dwWidth;
            d["height"] = header_info.header.dwHeight;
            d["depth"] = (header_info.header.dwFlags & DDSD_DEPTH) ? header_info.header.dwDepth : 1;
            d["format_str"] = DXGIFormatToString(header_info.format);
            d["mip_levels"] = (header_info.header.dwFlags & DDSD_MIPMAPCOUNT) ? header_info.header.dwMipMapCount : 1;
            d["array_size"] = (header_info.header.dwCaps2 & DDSCAPS2_CUBEMAP) ? 6 : 1;
            d["is_cubemap"] = (header_info.header.dwCaps2 & DDSCAPS2_CUBEMAP) != 0;
            d["is_3d"] = (header_info.header.dwCaps2 & DDSCAPS2_VOLUME) != 0;
            return d;
        } catch (const std::exception& e) {
            throw DDSLoadError(std::string("DDS metadata parsing failed on all paths. Fallback error: ") + e.what());
        }
    }

    nb::gil_scoped_acquire gil_acquire;
    nb::dict d;
    d["width"] = metadata.width;
    d["height"] = metadata.height;
    d["depth"] = metadata.depth;
    d["format_str"] = DXGIFormatToString(metadata.format);
    d["mip_levels"] = metadata.mipLevels;
    d["array_size"] = metadata.arraySize;
    d["is_cubemap"] = metadata.IsCubemap();
    d["is_3d"] = (metadata.dimension == TEX_DIMENSION_TEXTURE3D);
    return d;
}

// =======================================================================================
// DXGI Format to String Conversion
// =======================================================================================
std::string DXGIFormatToString(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_UNKNOWN: return "UNKNOWN";
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: return "R32G32B32A32_TYPELESS";
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_UINT: return "R32G32B32A32_UINT";
        case DXGI_FORMAT_R32G32B32A32_SINT: return "R32G32B32A32_SINT";
        case DXGI_FORMAT_R32G32B32_TYPELESS: return "R32G32B32_TYPELESS";
        case DXGI_FORMAT_R32G32B32_FLOAT: return "R32G32B32_FLOAT";
        case DXGI_FORMAT_R32G32B32_UINT: return "R32G32B32_UINT";
        case DXGI_FORMAT_R32G32B32_SINT: return "R32G32B32_SINT";
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return "R16G16B16A16_TYPELESS";
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R16G16B16A16_UNORM: return "R16G16B16A16_UNORM";
        case DXGI_FORMAT_R16G16B16A16_UINT: return "R16G16B16A16_UINT";
        case DXGI_FORMAT_R16G16B16A16_SNORM: return "R16G16B16A16_SNORM";
        case DXGI_FORMAT_R16G16B16A16_SINT: return "R16G16B16A16_SINT";
        case DXGI_FORMAT_R32G32_TYPELESS: return "R32G32_TYPELESS";
        case DXGI_FORMAT_R32G32_FLOAT: return "R32G32_FLOAT";
        case DXGI_FORMAT_R32G32_UINT: return "R32G32_UINT";
        case DXGI_FORMAT_R32G32_SINT: return "R32G32_SINT";
        case DXGI_FORMAT_R32G8X24_TYPELESS: return "R32G8X24_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return "D32_FLOAT_S8X24_UINT";
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return "R32_FLOAT_X8X24_TYPELESS";
        case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT: return "X32_TYPELESS_G8X24_UINT";
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return "R10G10B10A2_TYPELESS";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_R10G10B10A2_UINT: return "R10G10B10A2_UINT";
        case DXGI_FORMAT_R11G11B10_FLOAT: return "R11G11B10_FLOAT";
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return "R8G8B8A8_TYPELESS";
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_UINT: return "R8G8B8A8_UINT";
        case DXGI_FORMAT_R8G8B8A8_SNORM: return "R8G8B8A8_SNORM";
        case DXGI_FORMAT_R8G8B8A8_SINT: return "R8G8B8A8_SINT";
        case DXGI_FORMAT_R16G16_TYPELESS: return "R16G16_TYPELESS";
        case DXGI_FORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
        case DXGI_FORMAT_R16G16_UNORM: return "R16G16_UNORM";
        case DXGI_FORMAT_R16G16_UINT: return "R16G16_UINT";
        case DXGI_FORMAT_R16G16_SNORM: return "R16G16_SNORM";
        case DXGI_FORMAT_R16G16_SINT: return "R16G16_SINT";
        case DXGI_FORMAT_R32_TYPELESS: return "R32_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT: return "D32_FLOAT";
        case DXGI_FORMAT_R32_FLOAT: return "R32_FLOAT";
        case DXGI_FORMAT_R32_UINT: return "R32_UINT";
        case DXGI_FORMAT_R32_SINT: return "R32_SINT";
        case DXGI_FORMAT_R24G8_TYPELESS: return "R24G8_TYPELESS";
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return "D24_UNORM_S8_UINT";
        case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return "R24_UNORM_X8_TYPELESS";
        case DXGI_FORMAT_X24_TYPELESS_G8_UINT: return "X24_TYPELESS_G8_UINT";
        case DXGI_FORMAT_R8G8_TYPELESS: return "R8G8_TYPELESS";
        case DXGI_FORMAT_R8G8_UNORM: return "R8G8_UNORM";
        case DXGI_FORMAT_R8G8_UINT: return "R8G8_UINT";
        case DXGI_FORMAT_R8G8_SNORM: return "R8G8_SNORM";
        case DXGI_FORMAT_R8G8_SINT: return "R8G8_SINT";
        case DXGI_FORMAT_R16_TYPELESS: return "R16_TYPELESS";
        case DXGI_FORMAT_R16_FLOAT: return "R16_FLOAT";
        case DXGI_FORMAT_D16_UNORM: return "D16_UNORM";
        case DXGI_FORMAT_R16_UNORM: return "R16_UNORM";
        case DXGI_FORMAT_R16_UINT: return "R16_UINT";
        case DXGI_FORMAT_R16_SNORM: return "R16_SNORM";
        case DXGI_FORMAT_R16_SINT: return "R16_SINT";
        case DXGI_FORMAT_R8_TYPELESS: return "R8_TYPELESS";
        case DXGI_FORMAT_R8_UNORM: return "R8_UNORM";
        case DXGI_FORMAT_R8_UINT: return "R8_UINT";
        case DXGI_FORMAT_R8_SNORM: return "R8_SNORM";
        case DXGI_FORMAT_R8_SINT: return "R8_SINT";
        case DXGI_FORMAT_A8_UNORM: return "A8_UNORM";
        case DXGI_FORMAT_R1_UNORM: return "R1_UNORM";
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return "R9G9B9E5_SHAREDEXP";
        case DXGI_FORMAT_R8G8_B8G8_UNORM: return "R8G8_B8G8_UNORM";
        case DXGI_FORMAT_G8R8_G8B8_UNORM: return "G8R8_G8B8_UNORM";
        case DXGI_FORMAT_BC1_TYPELESS: return "BC1_TYPELESS";
        case DXGI_FORMAT_BC1_UNORM: return "BC1_UNORM";
        case DXGI_FORMAT_BC1_UNORM_SRGB: return "BC1_UNORM_SRGB";
        case DXGI_FORMAT_BC2_TYPELESS: return "BC2_TYPELESS";
        case DXGI_FORMAT_BC2_UNORM: return "BC2_UNORM";
        case DXGI_FORMAT_BC2_UNORM_SRGB: return "BC2_UNORM_SRGB";
        case DXGI_FORMAT_BC3_TYPELESS: return "BC3_TYPELESS";
        case DXGI_FORMAT_BC3_UNORM: return "BC3_UNORM";
        case DXGI_FORMAT_BC3_UNORM_SRGB: return "BC3_UNORM_SRGB";
        case DXGI_FORMAT_BC4_TYPELESS: return "BC4_TYPELESS";
        case DXGI_FORMAT_BC4_UNORM: return "BC4_UNORM";
        case DXGI_FORMAT_BC4_SNORM: return "BC4_SNORM";
        case DXGI_FORMAT_BC5_TYPELESS: return "BC5_TYPELESS";
        case DXGI_FORMAT_BC5_UNORM: return "BC5_UNORM";
        case DXGI_FORMAT_BC5_SNORM: return "BC5_SNORM";
        case DXGI_FORMAT_B5G6R5_UNORM: return "B5G6R5_UNORM";
        case DXGI_FORMAT_B5G5R5A1_UNORM: return "B5G5R5A1_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8X8_UNORM: return "B8G8R8X8_UNORM";
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: return "R10G10B10_XR_BIAS_A2_UNORM";
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return "B8G8R8A8_TYPELESS";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return "B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_B8G8R8X8_TYPELESS: return "B8G8R8X8_TYPELESS";
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return "B8G8R8X8_UNORM_SRGB";
        case DXGI_FORMAT_BC6H_TYPELESS: return "BC6H_TYPELESS";
        case DXGI_FORMAT_BC6H_UF16: return "BC6H_UF16";
        case DXGI_FORMAT_BC6H_SF16: return "BC6H_SF16";
        case DXGI_FORMAT_BC7_TYPELESS: return "BC7_TYPELESS";
        case DXGI_FORMAT_BC7_UNORM: return "BC7_UNORM";
        case DXGI_FORMAT_BC7_UNORM_SRGB: return "BC7_UNORM_SRGB";
        case DXGI_FORMAT_B4G4R4A4_UNORM: return "B4G4R4A4_UNORM";
        default: {
            std::ostringstream oss;
            oss << "FORMAT_CODE_0x" << std::hex << std::uppercase << static_cast<int>(format);
            return oss.str();
        }
    }
}

} // namespace

// =======================================================================================
// Python Module Definition
// =======================================================================================
NB_MODULE(directxtex_decoder, m) {
    m.doc() = "DDS decoder";

    m.def("decode_dds", &decode_dds,
          "Decodes a DDS file from bytes. Falls back to a manual parser for malformed files.",
          "dds_bytes"_a,
          "mip_level"_a = 0,
          "array_index"_a = -1, // -1 means load all array slices
          "force_rgba8"_a = false);

    m.def("get_dds_metadata", &get_dds_metadata,
          "Extracts DDS metadata without decoding pixel data. Uses a fallback for malformed files.",
          "dds_bytes"_a);

    // Register the custom exception to be catchable in Python
    static nb::exception<DDSLoadError> ex(m, "DDSLoadError");

    m.attr("__version__") = "1.0.0";
}
