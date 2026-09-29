"""
NVIDIA nvCOMP Python API type stubs.

Provides type information for the nvidia.nvcomp module, which offers
high-performance compression and decompression using the nvCOMP library.

Supported algorithms: LZ4, Snappy, Zstd, Cascaded, Deflate, GDeflate, ANS, Bitcomp.

See: https://docs.nvidia.com/cuda/nvcomp/py_api.html
"""

from collections.abc import Callable
from typing import Any, ClassVar, Self, overload

import numpy

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

COMPUTE_AND_NO_VERIFY: ChecksumPolicy
"""Compute checksums during compression but do not verify on decompression."""

COMPUTE_AND_VERIFY: ChecksumPolicy
"""Compute checksums during compression and verify on decompression."""

COMPUTE_AND_VERIFY_IF_PRESENT: ChecksumPolicy
"""Compute checksums during compression; verify on decompression only if present."""

NO_COMPUTE_AND_VERIFY_IF_PRESENT: ChecksumPolicy
"""Do not compute checksums; verify on decompression only if present."""

NO_COMPUTE_NO_VERIFY: ChecksumPolicy
"""Do not compute or verify checksums."""

NVCOMP_BITSHUFFLE_LSB_FIRST: BitshuffleMode
"""Bitshuffle with least-significant bit first ordering."""

NVCOMP_BITSHUFFLE_MSB_FIRST: BitshuffleMode
"""Bitshuffle with most-significant bit first ordering."""

NVCOMP_BITSHUFFLE_NONE: BitshuffleMode
"""No bitshuffle applied."""

NVCOMP_DECOMPRESS_BACKEND_CUDA: DecompressBackend
"""CUDA-based decompression backend."""

NVCOMP_DECOMPRESS_BACKEND_DEFAULT: DecompressBackend
"""Default decompression backend (nvCOMP decides the best backend)."""

NVCOMP_DECOMPRESS_BACKEND_HARDWARE: DecompressBackend
"""Hardware-based decompression backend."""

STRIDED_DEVICE: ArrayBufferKind
"""GPU-accessible buffer in pitch-linear layout."""

STRIDED_HOST: ArrayBufferKind
"""Host-accessible buffer in pitch-linear layout."""

__cuda_version__: int
"""The CUDA version nvCOMP was compiled against."""

__version__: str
"""The version string of the nvCOMP library."""

# ---------------------------------------------------------------------------
# Array
# ---------------------------------------------------------------------------

class Array:
    """Class which wraps an array buffer.

    Represents data to encode or decoded output.  Wraps both host and device
    buffers and provides interoperability via DLPack, ``__cuda_array_interface__``,
    and ``__array_interface__``.
    """

    def __init__(self, src_object: object, cuda_stream: int = ...) -> None:
        """Initialize an Array from an external buffer.

        Parameters:
            src_object: Input DLPack tensor (PyCapsule), or any object exposing
                ``__cuda_array_interface__``, ``__array_interface__``, or
                ``__dlpack__`` / ``__dlpack_device__``.
            cuda_stream: Optional ``cudaStream_t`` represented as a Python integer
                upon which synchronization must take place in the created Array.
        """
        ...

    def cpu(self) -> Self:
        """Return a copy of this array in CPU memory.

        If this array is already in CPU memory, no copy is performed and the
        original object is returned.

        Returns:
            Array object with content in CPU memory, or ``None`` if copy could
            not be done.
        """
        ...

    def cuda(self, synchronize: bool = True, cuda_stream: int = 0) -> Self:
        """Return a copy of this array in device memory.

        If this array is already in device memory, no copy is performed and the
        original object is returned.

        Parameters:
            synchronize: If ``True`` (default), block until the host-to-device
                copy completes.  If ``False``, no synchronization is executed;
                further synchronization must be done using the CUDA stream
                provided e.g. via ``__cuda_array_interface__``.
            cuda_stream: Optional ``cudaStream_t`` represented as a Python
                integer to copy the host buffer to.

        Returns:
            Array object with content in device memory, or ``None`` if copy
            could not be done.
        """
        ...

    def to_dlpack(self, cuda_stream: int | None = None) -> Any:
        """Export the array with zero-copy conversion to a DLPack tensor.

        Parameters:
            cuda_stream: Optional ``cudaStream_t`` represented as a Python
                integer upon which synchronization must take place in the
                created Array.

        Returns:
            DLPack tensor encapsulated in a ``PyCapsule`` object.
        """
        ...

    def __buffer__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __dlpack__(self, stream: CudaStream | None = None) -> Any:
        """Export the array as a DLPack tensor.

        Parameters:
            stream: Optional stream for synchronization.

        Returns:
            DLPack tensor encapsulated in a ``PyCapsule`` object.
        """
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        """Get the device associated with the buffer.

        Returns:
            A tuple ``(device_type, device_id)`` following the DLPack convention.
        """
        ...

    def __release_buffer__(self, *args: Any, **kwargs: Any) -> Any: ...
    @property
    def buffer_kind(self) -> ArrayBufferKind:
        """Buffer kind in which array data is stored (host or device)."""
        ...

    @property
    def buffer_size(self) -> int:
        """The total number of bytes to store the array."""
        ...

    @property
    def capacity(self) -> int:
        """Amount of memory allocated for the array."""
        ...

    @property
    def dtype(self) -> numpy.dtype:
        """The NumPy data type of the array elements."""
        ...

    @property
    def item_size(self) -> int:
        """Size of each element in bytes."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        ...

    @property
    def precision(self) -> int:
        """Maximum number of significant bits in the data type.

        A value of 0 means that precision is equal to the data type bit depth.
        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array as a tuple of dimension sizes."""
        ...

    @property
    def size(self) -> int:
        """Total number of elements this array holds."""
        ...

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides of axes in bytes."""
        ...

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """NumPy ``__array_interface__`` protocol for host buffers."""
        ...

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        """CUDA array interchange interface compatible with Numba v0.39.0 or later."""
        ...

# ---------------------------------------------------------------------------
# ArrayBufferKind
# ---------------------------------------------------------------------------

class ArrayBufferKind:
    """Defines the buffer kind in which array data is stored.

    Members:
        STRIDED_DEVICE: GPU-accessible in pitch-linear layout.
        STRIDED_HOST: Host-accessible in pitch-linear layout.
    """

    __members__: ClassVar[dict[str, ArrayBufferKind]] = ...  # read-only
    STRIDED_DEVICE: ClassVar[ArrayBufferKind] = ...
    """GPU-accessible in pitch-linear layout."""
    STRIDED_HOST: ClassVar[ArrayBufferKind] = ...
    """Host-accessible in pitch-linear layout."""
    __entries: ClassVar[dict[str, ArrayBufferKind]] = ...

    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str:
        """The name of this buffer kind."""
        ...

    @property
    def value(self) -> int:
        """The integer value of this buffer kind."""
        ...

# ---------------------------------------------------------------------------
# BitshuffleMode
# ---------------------------------------------------------------------------

class BitshuffleMode:
    """Defines the bitshuffle mode for LZ4 and Cascaded algorithms.

    Members:
        NVCOMP_BITSHUFFLE_NONE: No bitshuffle applied.
        NVCOMP_BITSHUFFLE_MSB_FIRST: Most-significant bit first ordering.
        NVCOMP_BITSHUFFLE_LSB_FIRST: Least-significant bit first ordering.
    """

    __members__: ClassVar[dict[str, BitshuffleMode]] = ...  # read-only
    NVCOMP_BITSHUFFLE_LSB_FIRST: ClassVar[BitshuffleMode] = ...
    """Bitshuffle with least-significant bit first ordering."""
    NVCOMP_BITSHUFFLE_MSB_FIRST: ClassVar[BitshuffleMode] = ...
    """Bitshuffle with most-significant bit first ordering."""
    NVCOMP_BITSHUFFLE_NONE: ClassVar[BitshuffleMode] = ...
    """No bitshuffle applied."""
    __entries: ClassVar[dict[str, BitshuffleMode]] = ...

    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str:
        """The name of this bitshuffle mode."""
        ...

    @property
    def value(self) -> int:
        """The integer value of this bitshuffle mode."""
        ...

# ---------------------------------------------------------------------------
# BitstreamKind
# ---------------------------------------------------------------------------

class BitstreamKind:
    """Defines how a buffer will be compressed in nvCOMP.

    Members:
        NVCOMP_NATIVE: Each input buffer is chunked according to manager setting
            and compressed in parallel.  Allows computation of checksums.  Adds
            a custom header with nvCOMP metadata at the beginning of the
            compressed data.
        RAW: Compresses input data as-is using the underlying compression
            algorithm.  Does not add a header with nvCOMP metadata.
        WITH_UNCOMPRESSED_SIZE: Similar to ``RAW``, but adds a custom header
            with just the uncompressed size at the beginning of the compressed
            data.
    """

    __members__: ClassVar[dict[str, BitstreamKind]] = ...  # read-only
    NVCOMP_NATIVE: ClassVar[BitstreamKind] = ...
    """Chunked parallel compression with nvCOMP metadata header."""
    RAW: ClassVar[BitstreamKind] = ...
    """Raw compression without nvCOMP metadata header."""
    WITH_UNCOMPRESSED_SIZE: ClassVar[BitstreamKind] = ...
    """Raw compression with uncompressed size prefix header."""
    __entries: ClassVar[dict[str, BitstreamKind]] = ...

    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str:
        """The name of this bitstream kind."""
        ...

    @property
    def value(self) -> int:
        """The integer value of this bitstream kind."""
        ...

# ---------------------------------------------------------------------------
# ChecksumPolicy
# ---------------------------------------------------------------------------

class ChecksumPolicy:
    """Defines the strategy for computing and verifying checksums.

    Members:
        NO_COMPUTE_NO_VERIFY: Do not compute or verify checksums.
        COMPUTE_AND_NO_VERIFY: Compute checksums but do not verify.
        COMPUTE_AND_VERIFY: Compute and verify checksums.
        COMPUTE_AND_VERIFY_IF_PRESENT: Compute checksums; verify only if present.
        NO_COMPUTE_AND_VERIFY_IF_PRESENT: Do not compute; verify only if present.
    """

    __members__: ClassVar[dict[str, ChecksumPolicy]] = ...  # read-only
    COMPUTE_AND_NO_VERIFY: ClassVar[ChecksumPolicy] = ...
    """Compute checksums during compression but do not verify on decompression."""
    COMPUTE_AND_VERIFY: ClassVar[ChecksumPolicy] = ...
    """Compute checksums during compression and verify on decompression."""
    COMPUTE_AND_VERIFY_IF_PRESENT: ClassVar[ChecksumPolicy] = ...
    """Compute checksums; verify on decompression only if present."""
    NO_COMPUTE_AND_VERIFY_IF_PRESENT: ClassVar[ChecksumPolicy] = ...
    """Do not compute checksums; verify on decompression only if present."""
    NO_COMPUTE_NO_VERIFY: ClassVar[ChecksumPolicy] = ...
    """Do not compute or verify checksums."""
    __entries: ClassVar[dict[str, ChecksumPolicy]] = ...

    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str:
        """The name of this checksum policy."""
        ...

    @property
    def value(self) -> int:
        """The integer value of this checksum policy."""
        ...

# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

class Codec:
    """High-level compression / decompression codec.

    Wraps the nvCOMP compression manager and provides ``encode`` / ``decode``
    for single or batched arrays.  Supports LZ4, Snappy, Zstd, Cascaded,
    Deflate, GDeflate, ANS, and Bitcomp algorithms.

    Thread safety: each thread must create and use its own ``Codec`` instance.
    A single ``Codec`` must not be called from multiple threads concurrently
    because it holds mutable internal state.
    """

    def __init__(
        self,
        *,
        algorithm: str = "",
        device_id: int = ...,
        cuda_stream: int = ...,
        uncomp_chunk_size: int = 65536,
        bitstream_kind: BitstreamKind = BitstreamKind.NVCOMP_NATIVE,
        checksum_policy: ChecksumPolicy = NO_COMPUTE_NO_VERIFY,
        decompress_backend: DecompressBackend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT,
        use_de_sort: bool = ...,
        # LZ4 / Cascaded specific
        bitshuffle_mode: BitshuffleMode = BitshuffleMode.NVCOMP_BITSHUFFLE_NONE,
        # GDeflate / Deflate / Gzip specific
        algorithm_type: int = ...,
        # ANS / LZ4 / Cascaded specific
        data_type: str = ...,
        # Cascaded specific
        num_rles: int = 2,
        num_deltas: int = 1,
        use_bitpack: bool = True,
    ) -> None:
        """Initialize the codec.

        Parameters:
            algorithm: Optional name of the compression algorithm to use
                (e.g. ``"LZ4"``, ``"Snappy"``, ``"Zstd"``, ``"Cascaded"``,
                ``"Deflate"``, ``"GDeflate"``, ``"ANS"``, ``"Bitcomp"``).
                By default it is empty and the algorithm can be deduced during
                decoding.
            device_id: Optional device ID for encoding / decoding.  If not
                specified the default (current) device is used.
            cuda_stream: Optional ``cudaStream_t`` as a Python integer.  By
                default an internal CUDA stream is created for the given device.
            uncomp_chunk_size: Optional uncompressed data chunk size in bytes.
                Default is 65536 (64 KiB).
            bitstream_kind: Format of the bitstream nvCOMP will work on.
                Default is ``BitstreamKind.NVCOMP_NATIVE``.
            checksum_policy: Strategy for computing / verifying checksums.
                Default is ``ChecksumPolicy.NO_COMPUTE_NO_VERIFY``.
            decompress_backend: Decompression strategy (HW / CUDA).  Default is
                ``DecompressBackend.NVCOMP_DECOMPRESS_BACKEND_DEFAULT``.  Used
                for LZ4, Snappy, Deflate, and Gzip algorithms.
            use_de_sort: Whether to sort chunks before hardware decompression
                for better performance.  Only used when the backend is
                ``NVCOMP_DECOMPRESS_BACKEND_HARDWARE``.
            bitshuffle_mode: (LZ4 / Cascaded) Optional bitshuffle mode.
                Default is ``BitshuffleMode.NVCOMP_BITSHUFFLE_NONE``.
            algorithm_type: (GDeflate / Deflate / Gzip / Bitcomp) Compression
                algorithm type.  For GDeflate/Deflate/Gzip: 0–5 (0 = highest
                throughput, 5 = highest compression ratio; 1 is default).
                For Bitcomp: 0 = default, 1 = sparse.
            data_type: (LZ4 / ANS / Cascaded) Optional array-protocol type
                string for the default data type.
            num_rles: (Cascaded) Number of Run Length Encodings.  Default 2.
            num_deltas: (Cascaded) Number of Delta Encodings.  Default 1.
            use_bitpack: (Cascaded) Whether to bitpack the final layers.
                Default ``True``.
        """
        ...

    # -- compression_config --------------------------------------------------

    @overload
    def compression_config(self, uncompressed_size: int) -> CompressConfig:
        """Build a compression config from the uncompressed buffer size.

        Unlike ``decompression_config`` this does **not** synchronize the
        codec's CUDA stream.

        Parameters:
            uncompressed_size: Uncompressed buffer size in bytes.  Must be
                greater than zero.

        Returns:
            A new ``CompressConfig``.
        """
        ...

    @overload
    def compression_config(self, uncompressed_sizes: list[int]) -> CompressConfig:
        """Build a batch compression config from per-element sizes.

        Unlike ``decompression_config`` this does **not** synchronize the
        codec's CUDA stream.

        Parameters:
            uncompressed_sizes: List of uncompressed buffer sizes in bytes.
                Must be non-empty; each entry must be greater than zero.

        Returns:
            A new ``CompressConfig``.
        """
        ...

    # -- decompression_config ------------------------------------------------

    @overload
    def decompression_config(self, src: Array) -> DecompressConfig:
        """Build a decompression config by parsing the compressed buffer header.

        .. warning::
            This call synchronizes the codec's CUDA stream to read the
            compressed buffer's metadata back to the host.

        The returned ``DecompressConfig`` can be reused across multiple
        ``decode`` calls, as long as the compressed buffers have the same
        uncompressed shape.

        Parameters:
            src: Encoded ``Array``.

        Returns:
            A new ``DecompressConfig``.
        """
        ...

    @overload
    def decompression_config(self, srcs: list[Array]) -> DecompressConfig:
        """Build a batch decompression config by parsing compressed buffer headers.

        .. warning::
            This call synchronizes the codec's CUDA stream to read each
            compressed buffer's metadata back to the host.

        The returned ``DecompressConfig`` can be reused across multiple
        ``decode`` calls, as long as the compressed buffers have the same
        uncompressed per-element shape.

        Parameters:
            srcs: List of encoded ``Array`` objects.

        Returns:
            A new ``DecompressConfig``.
        """
        ...

    @overload
    def decompression_config(
        self, compression_config: CompressConfig
    ) -> DecompressConfig:
        """Build a decompression config directly from a compression config.

        The resulting ``DecompressConfig`` is reusable across any compressed
        buffer produced by ``encode`` calls that used the same
        ``CompressConfig`` on the same ``Codec`` object (i.e. same
        uncompressed shape).  This is the recommended path when compressing
        and decompressing in the same process — combine with
        ``compression_config(size)`` for a fully sync-free round trip.

        Parameters:
            compression_config: A ``CompressConfig`` previously obtained from
                ``codec.compression_config(…)``.

        Returns:
            A new ``DecompressConfig``.
        """
        ...

    # -- decode --------------------------------------------------------------

    @overload
    def decode(
        self,
        src: Array,
        data_type: str = ...,
        out: Array | None = None,
        decompression_config: DecompressConfig | None = None,
    ) -> Array:
        """Decode (decompress) a single ``Array``.

        Parameters:
            src: Decode source ``Array``.
            data_type: Optional array-protocol type string for the output data
                type.  Default is ``"|u1"``.
            out: Optional writable buffer to store decoded data.  If it is a
                native ``nvcomp.Array`` it will be resized to fit the
                decompressed output.  If it is an externally-allocated buffer
                (e.g. cupy / numba array) its size is fixed and ``ValueError``
                is raised when it is too small.
            decompression_config: Optional config from
                ``codec.decompression_config``.  Three cases:

                * Not provided — ``decode`` internally calls
                  ``configure_decompression`` on *src*, forcing a stream
                  synchronization.
                * Provided, built from a buffer via
                  ``codec.decompression_config(compressed)`` — the
                  synchronization already happened when the config was built.
                * Provided, built from a ``CompressConfig`` via
                  ``codec.decompression_config(comp_cfg)`` — no header parse
                  ever happened; fully sync-free (recommended pattern).

                The same ``DecompressConfig`` can be reused across multiple
                ``decode`` calls on different compressed buffers of the same
                uncompressed shape.

        Returns:
            Decoded ``nvcomp.Array``.
        """
        ...

    @overload
    def decode(
        self,
        srcs: list[Array],
        data_type: str = ...,
        out: list[Array] | None = None,
        decompression_config: DecompressConfig | None = None,
    ) -> list[Array]:
        """Decode (decompress) a batch of ``Array`` objects.

        Parameters:
            srcs: List of ``Array`` objects to decode.
            data_type: Optional array-protocol type string for the output data
                type.
            out: Optional list of writable arrays to store decoded data.  Each
                native ``nvcomp.Array`` entry will be resized to fit.
                Externally-allocated buffers keep their fixed size and
                ``ValueError`` is raised when any is too small.
            decompression_config: Optional config — see the single-array
                overload for details.

        Returns:
            List of decoded ``nvcomp.Array`` objects.
        """
        ...

    # -- encode --------------------------------------------------------------

    @overload
    def encode(
        self,
        array: Array,
        out: Array | None = None,
        compression_config: CompressConfig | None = None,
    ) -> Array:
        """Encode (compress) a single ``Array``.

        Parameters:
            array: ``Array`` to encode.
            out: Optional writable buffer to store encoded data.  If it is a
                native ``nvcomp.Array`` it will be resized to fit the
                compressed output.  If it is an externally-allocated buffer
                (e.g. cupy / numba array) its size is fixed and ``ValueError``
                is raised when it is too small.
            compression_config: Optional config from
                ``codec.compression_config``.  When provided, ``encode`` skips
                the internal ``configure_compression`` step.

        Returns:
            Encoded ``nvcomp.Array``.
        """
        ...

    @overload
    def encode(
        self,
        srcs: list[Array],
        out: list[Array] | None = None,
        compression_config: CompressConfig | None = None,
    ) -> list[Array]:
        """Encode (compress) a batch of ``Array`` objects.

        As in the single-array overload, ``configure_compression`` does not
        synchronize the stream, so this call is fully asynchronous regardless
        of whether *compression_config* is provided.

        Parameters:
            srcs: List of ``Array`` objects to encode.
            out: Optional list of writable arrays to store encoded data.  Each
                native ``nvcomp.Array`` entry will be resized to fit.
                Externally-allocated buffers keep their fixed size and
                ``ValueError`` is raised when any is too small.
            compression_config: Optional config from
                ``codec.compression_config``.  When provided, ``encode`` skips
                the internal configure step.

        Returns:
            List of encoded ``nvcomp.Array`` objects.
        """
        ...

    # -- buffer-size helpers -------------------------------------------------

    def get_max_comp_buffer_size(self, source: Array) -> int:
        """Retrieve the maximum compressed buffer size for an uncompressed array.

        Returns an upper bound on the number of bytes the codec may emit when
        compressing an input of the given size.  Use this to pre-allocate an
        output buffer that is guaranteed to fit the compressed result.  The
        exact compressed size is typically smaller and is only known after
        encoding completes.

        Parameters:
            source: Input ``Array`` object.

        Returns:
            Upper bound in bytes for the compressed output.
        """
        ...

    def get_uncomp_buffer_size(self, source: Array) -> int:
        """Retrieve the uncompressed buffer size from a compressed nvCOMP array.

        The size is read from metadata embedded in the compressed bitstream and
        depends on the codec's ``bitstream_kind``:

        * ``NVCOMP_NATIVE``: read from the nvCOMP common header.
        * ``WITH_UNCOMPRESSED_SIZE``: read from the size prefix at the start
          of the buffer.
        * ``RAW``: derived by calling each format's native size-query path.
          Bitcomp, Zstd, Snappy, ANS, and Cascaded perform a constant-time
          lookup; LZ4, Deflate, and GDeflate walk the compressed stream.

        Both host and device buffers are accepted; for device buffers this call
        synchronizes the codec's CUDA stream before returning.

        Parameters:
            source: Input ``Array`` containing nvCOMP compressed data produced
                with a matching algorithm and bitstream kind.

        Returns:
            Size in bytes of the uncompressed payload.
        """
        ...

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> Codec:
        """Enter the runtime context (returns *self*)."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc_value: BaseException | None = ...,
        traceback: Any | None = ...,
    ) -> None:
        """Exit the runtime context and release resources."""
        ...

# ---------------------------------------------------------------------------
# CompressConfig
# ---------------------------------------------------------------------------

class CompressConfig:
    """Configuration for compression, built via ``Codec.compression_config``.

    Can be passed to ``Codec.encode`` to skip the internal
    ``configure_compression`` step, and to ``Codec.decompression_config`` to
    build a ``DecompressConfig`` without any stream synchronization.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# ---------------------------------------------------------------------------
# CudaStream
# ---------------------------------------------------------------------------

class CudaStream:
    """Wrapper around a CUDA stream.

    Provides either shared-ownership or view semantics, depending on whether it
    was constructed through ``make_new`` or ``borrow``, respectively.

    ``CudaStream`` is the type of stream parameters passed to allocation
    functions that can be used with ``set_*_allocator``.  If deallocation of
    such memory needs to access the stream passed to the allocation function,
    the allocation function should return an ``ExternalMemory`` instance
    wrapping the newly constructed memory object and the ``CudaStream``
    argument.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def borrow(cuda_stream: int, device_idx: int = -1) -> CudaStream:
        """Create a stream view (non-owning wrapper).

        The device index is primarily intended for special CUDA streams (i.e.
        the default, legacy, and per-thread streams) whose device cannot be
        inferred from the stream value itself.  By default it is -1, a special
        value whose meaning depends on whether *cuda_stream* is special or not.
        If *cuda_stream* is special, the default value associates the shared
        stream with the current device.  Otherwise the ``CudaStream`` will
        always be associated with the stream's actual device; passing a
        *device_idx* that is neither the default nor the stream's actual device
        raises an exception.

        Parameters:
            cuda_stream: The ``cudaStream_t`` to wrap, represented as a Python
                integer.
            device_idx: Optional index of the device with which to associate
                the borrowed stream.  Default -1.

        Returns:
            A new ``CudaStream`` with view semantics.
        """
        ...

    @staticmethod
    def make_new(device_idx: int = -1) -> CudaStream:
        """Create a new CUDA stream with shared ownership.

        Parameters:
            device_idx: Optional index of the device with which to associate
                the newly created stream.  Default -1 (current device).

        Returns:
            A new ``CudaStream`` with shared-ownership semantics.
        """
        ...

    @property
    def device(self) -> int:
        """The device index associated with the stream."""
        ...

    @property
    def is_special(self) -> bool:
        """Whether the underlying stream is one of the special CUDA streams.

        Special streams are the default, legacy, or per-thread default streams.
        Passing a special stream to any CUDA API call will actually use the
        current device's corresponding special stream.  It is the user's
        responsibility to ensure that the stream's associated device (as given
        by ``device``) is selected before using the stream.
        """
        ...

    @property
    def ptr(self) -> int:
        """The underlying ``cudaStream_t`` represented as a Python integer.

        The property name follows the convention of ``cupy.Stream`` and
        reflects the fact that a ``cudaStream_t`` is internally a pointer.
        """
        ...

# ---------------------------------------------------------------------------
# DecompressBackend
# ---------------------------------------------------------------------------

class DecompressBackend:
    """Defines the decompression strategy (hardware vs. CUDA-based).

    Members:
        NVCOMP_DECOMPRESS_BACKEND_DEFAULT: Let nvCOMP decide the best backend.
        NVCOMP_DECOMPRESS_BACKEND_CUDA: Use the CUDA-based decompression.
        NVCOMP_DECOMPRESS_BACKEND_HARDWARE: Use the hardware decompression.
    """

    __members__: ClassVar[dict[str, DecompressBackend]] = ...  # read-only
    NVCOMP_DECOMPRESS_BACKEND_CUDA: ClassVar[DecompressBackend] = ...
    """CUDA-based decompression backend."""
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT: ClassVar[DecompressBackend] = ...
    """Default backend — nvCOMP decides the best strategy."""
    NVCOMP_DECOMPRESS_BACKEND_HARDWARE: ClassVar[DecompressBackend] = ...
    """Hardware-based decompression backend."""
    __entries: ClassVar[dict[str, DecompressBackend]] = ...

    def __init__(self, value: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    @property
    def name(self) -> str:
        """The name of this decompress backend."""
        ...

    @property
    def value(self) -> int:
        """The integer value of this decompress backend."""
        ...

# ---------------------------------------------------------------------------
# DecompressConfig
# ---------------------------------------------------------------------------

class DecompressConfig:
    """Configuration for decompression, built via ``Codec.decompression_config``.

    Can be passed to ``Codec.decode`` to avoid redundant header parsing and
    stream synchronization.  Reusable across multiple ``decode`` calls as long
    as the compressed buffers share the same uncompressed shape.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def as_array(source: object, cuda_stream: int = 0) -> Array:
    """Wrap an external buffer as an ``Array``.

    Ties the buffer lifetime to the returned array.

    Parameters:
        source: Input DLPack tensor (``PyCapsule``), or any object with
            ``__cuda_array_interface__``, ``__array_interface__``, or
            ``__dlpack__`` / ``__dlpack_device__`` methods.
        cuda_stream: Optional ``cudaStream_t`` as a Python integer upon which
            synchronization must take place in the created ``Array``.

    Returns:
        An ``nvcomp.Array`` wrapping *source*.
    """
    ...

def as_arrays(sources: list[object], cuda_stream: int = 0) -> list[Array]:
    """Wrap external buffers as ``Array`` objects.

    Ties each buffer's lifetime to its corresponding array.

    Parameters:
        sources: List of input DLPack tensors (``PyCapsule`` objects), or other
            objects with ``__cuda_array_interface__``, ``__array_interface__``,
            or ``__dlpack__`` / ``__dlpack_device__`` methods.
        cuda_stream: Optional ``cudaStream_t`` as a Python integer upon which
            synchronization must take place in the created ``Array`` objects.

    Returns:
        List of ``nvcomp.Array`` objects wrapping each element of *sources*.
    """
    ...

def from_dlpack(source: object, cuda_stream: int = 0) -> Array:
    """Zero-copy conversion from a DLPack tensor to an ``Array``.

    Parameters:
        source: Input DLPack tensor (``PyCapsule``), or any (array) object with
            ``__dlpack__`` and ``__dlpack_device__`` methods.
        cuda_stream: Optional ``cudaStream_t`` as a Python integer upon which
            synchronization must take place in the created ``Array``.

    Returns:
        An ``nvcomp.Array`` wrapping the DLPack tensor.
    """
    ...

def set_device_allocator(
    allocator: Callable[[int, CudaStream], object] | None = ...,
) -> None:
    """Set the allocator for future device (GPU) allocations.

    The allocator signature is::

        def my_allocator(nbytes: int, stream: CudaStream) -> PtrProtocol: ...

    where ``PtrProtocol`` is any object with a ``ptr`` attribute of integral
    type (the device pointer).  When the returned object is deleted the memory
    must be deallocated.

    The allocated memory must be device-accessible.

    Parameters:
        allocator: Callable satisfying the conditions above, or ``None`` to
            restore the default allocator.
    """
    ...

def set_host_allocator(
    allocator: Callable[[int, CudaStream], object] | None = ...,
) -> None:
    """Set the allocator for future non-pinned host allocations.

    This is primarily intended for potentially large allocations such as those
    backing CPU ``Array`` instances.  Moderately-sized internal host
    allocations may still use system-allocated memory.

    For pinned host memory use ``set_pinned_allocator`` instead.

    Parameters:
        allocator: Callable satisfying the same conditions as
            ``set_device_allocator``, but must allocate host-accessible memory.
            Pass ``None`` to restore the default allocator.
    """
    ...

def set_pinned_allocator(
    allocator: Callable[[int, CudaStream], object] | None = ...,
) -> None:
    """Set the allocator for future pinned host allocations.

    The allocator must allocate host-accessible (pinned) memory.  For
    non-pinned host memory use ``set_host_allocator``.

    Parameters:
        allocator: Callable satisfying the same conditions as
            ``set_device_allocator``, but must allocate host-accessible
            (preferably pinned) memory.  Pass ``None`` to restore the default
            allocator.
    """
    ...
