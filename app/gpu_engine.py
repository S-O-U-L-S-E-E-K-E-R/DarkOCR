"""The GPU text engine: PP-OCR models as TensorRT engines, with the work around them on the GPU.

The model weights are the stock PP-OCRv6 files that RapidOCR downloads. Three things differ
from running them through RapidOCR's own TensorRT support, and together they make reading
about twice as fast with the same text output:

  det  takes the resized uint8 image as it is. Scaling to [-1, 1] and the channel reorder,
       which cost about 30 ms per photo in NumPy, are layers inside the engine.
  rec  returns, per time step, the best character index and its probability, instead of the
       full table (18,710 classes per time step) that then has to be searched on the CPU.
  cls  runs on the GPU in full precision. In half precision it lost about 10% of the text.

RapidOCR still does everything that is not a bottleneck: box extraction, cropping, CTC
decoding and ordering. Engines are built on first use for this GPU and TensorRT version.
"""

import os
import re
import subprocess

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(os.path.dirname(APP_DIR), "engines")
REC_BATCH = 6        # as RapidOCR's default, so line padding and results stay the same
CLS_BATCH = 128     # all lines of an image in one call: its result per line does not depend on the batch
MAX_ASPECT = 2.5     # longest side / shortest side the detection engine can take
TUNED_LEVEL = 5      # TensorRT optimisation level of the optional tuned detection engine


# ---- model files and the changes made to them ------------------------------------------

def model_file(name):
    import rapidocr
    return os.path.join(os.path.dirname(rapidocr.__file__), "models", name)


def det_model():
    """uint8 [1, H, W, 3]  ->  probability map [1, 1, H, W]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    model = onnx.load(model_file("PP-OCRv6_det_small.onnx"))
    graph = model.graph
    old_input = graph.input[0].name
    graph.initializer.extend(numpy_helper.from_array(np.array(value, np.float32), name)
                             for name, value in (("pre_scale", 1 / 255.0), ("pre_mean", 0.5), ("pre_std", 0.5)))
    nodes = [
        helper.make_node("Cast", ["image"], ["pre_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["pre_float", "pre_scale"], ["pre_scaled"]),
        helper.make_node("Sub", ["pre_scaled", "pre_mean"], ["pre_centered"]),
        helper.make_node("Div", ["pre_centered", "pre_std"], ["pre_normalized"]),
        helper.make_node("Transpose", ["pre_normalized"], [old_input], perm=[0, 3, 1, 2]),
    ]
    for position, node in enumerate(nodes):
        graph.node.insert(position, node)
    del graph.input[:]
    graph.input.append(helper.make_tensor_value_info("image", TensorProto.UINT8, [1, "height", "width", 3]))
    return model


def rec_model():
    """float [N, 3, 48, W]  ->  best class index [N, T] and its probability [N, T]."""
    import onnx
    from onnx import TensorProto, helper
    model = onnx.load(model_file("PP-OCRv6_rec_small.onnx"))
    graph = model.graph
    table = graph.output[0].name
    graph.node.extend([
        helper.make_node("ArgMax", [table], ["best_index"], axis=2, keepdims=0),
        helper.make_node("ReduceMax", [table], ["best_probability"], axes=[2], keepdims=0),
    ])
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info("best_index", TensorProto.INT64, ["batch", "steps"]),
        helper.make_tensor_value_info("best_probability", TensorProto.FLOAT, ["batch", "steps"]),
    ])
    return model


def cls_model():
    """float [N, 3, 48, 192]  ->  [N, 2] probabilities for 0 and 180 degrees."""
    import onnx
    from onnx import TensorProto, helper
    model = onnx.load(model_file("ch_ppocr_mobile_v2.0_cls_mobile.onnx"))
    # The file names height and width with the same symbol, which TensorRT reads as
    # "height equals width". The pipeline always feeds 48 x 192, so state that.
    name = model.graph.input[0].name
    del model.graph.input[:]
    model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 3, 48, 192]))
    return model


# name: (model builder, half precision, (min, opt, max) input shapes)
ENGINES = {
    "det": (det_model, True, ((1, 32, 32, 3), (1, 1120, 1984, 3), (1, 2048, 2048, 3))),
    "rec": (rec_model, True, ((1, 3, 48, 32), (REC_BATCH, 3, 48, 320), (REC_BATCH, 3, 48, 4800))),
    "cls": (cls_model, False, ((1, 3, 48, 192), (48, 3, 48, 192), (CLS_BATCH, 3, 48, 192))),
}
# Added to an engine's file name when its settings change, so that an updated program builds
# the new engine instead of loading a file built for the old settings.
REVISIONS = {"cls": "-r2"}


# ---- building ---------------------------------------------------------------------------

def machine_tag():
    """Engines only work on the GPU model and TensorRT version they were built with."""
    import tensorrt as trt
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        gpu = "gpu"
    return re.sub(r"[^a-z0-9]+", "-", f"{gpu}-trt{trt.__version__}".lower()).strip("-")


def engine_path(name, tuned=False):
    stem = name + REVISIONS.get(name, "") + ("-tuned" if tuned else "")
    return os.path.join(ENGINE_DIR, f"{stem}.{machine_tag()}.engine")


def build(name, tuned=False):
    """Build one engine file. tuned=True searches much longer for the fastest kernels."""
    import tensorrt as trt
    make_model, half, shapes = ENGINES[name]
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(make_model().SerializeToString()):
        raise RuntimeError(f"{name}: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if tuned:
        config.builder_optimization_level = TUNED_LEVEL
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, *shapes)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"{name}: engine build failed")
    os.makedirs(ENGINE_DIR, exist_ok=True)
    path = engine_path(name, tuned)
    with open(path + ".part", "wb") as handle:
        handle.write(serialized)
    os.replace(path + ".part", path)
    return path


def ensure_engines():
    for name in ENGINES:
        if not os.path.exists(engine_path(name)):
            print(f"Building the {name} GPU engine for this machine (once) ...", flush=True)
            build(name)


def tune():
    """Build the tuned detection engine. Takes 10 to 30 minutes; used automatically afterwards."""
    import onnxruntime
    onnxruntime.preload_dlls()
    return build("det", tuned=True)


# ---- running ----------------------------------------------------------------------------

class Runner:
    """One TensorRT engine: one input, any number of outputs, dynamic shapes."""

    def __init__(self, engine_file):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.cudart = cudart
        self.to_device = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        self.to_host = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        with open(engine_file, "rb") as handle:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"cannot load {engine_file}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_input = lambda name: self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        self.input = next(name for name in names if is_input(name))
        self.outputs = [name for name in names if not is_input(name)]
        self.input_type = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input)))
        self.output_types = [np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))) for name in self.outputs]
        self.max_shape = tuple(self.engine.get_tensor_profile_shape(self.input, 0)[2])
        self.shape = None
        self.output_shapes = None
        self.capacity = {}  # tensor name -> bytes allocated on the GPU
        self.device = {}    # tensor name -> GPU pointer
        self._reserve(self.input, int(np.prod(self.max_shape)) * self.input_type.itemsize)

    def _check(self, result):
        if result[0].value != 0:
            raise RuntimeError(f"CUDA error: {result[0]}")
        return result[1] if len(result) > 1 else None

    def _reserve(self, name, size):
        if self.capacity.get(name, 0) < size:
            if name in self.device:
                self._check(self.cudart.cudaFree(self.device[name]))
            self.device[name] = self._check(self.cudart.cudaMalloc(size))
            self.capacity[name] = size
            self.context.set_tensor_address(name, self.device[name])

    def __call__(self, array):
        """Run the engine. Returns one NumPy array per output."""
        if array.dtype != self.input_type or not array.flags.c_contiguous:
            array = np.ascontiguousarray(array, self.input_type)
        if array.shape != self.shape:
            # TensorRT returns garbage, without an error, for shapes outside its range.
            if len(array.shape) != len(self.max_shape) or any(a > b for a, b in zip(array.shape, self.max_shape)):
                raise ValueError(f"input shape {array.shape} is outside the engine range {self.max_shape}")
            if not self.context.set_input_shape(self.input, array.shape):
                raise ValueError(f"engine refused input shape {array.shape}")
            self.shape = array.shape
            self.output_shapes = [tuple(self.context.get_tensor_shape(name)) for name in self.outputs]
            for name, shape, kind in zip(self.outputs, self.output_shapes, self.output_types):
                self._reserve(name, int(np.prod(shape)) * kind.itemsize)
        self._check(self.cudart.cudaMemcpy(self.device[self.input], array.ctypes.data, array.nbytes, self.to_device))
        if not self.context.execute_async_v3(0):
            raise RuntimeError("TensorRT execution failed")
        results = []
        for name, shape, kind in zip(self.outputs, self.output_shapes, self.output_types):
            result = np.empty(shape, kind)
            self._check(self.cudart.cudaMemcpy(result.ctypes.data, self.device[name], result.nbytes, self.to_host))
            results.append(result)
        return results


class DetPreProcess:
    """Resize only; the engine normalises. Same resize rule as RapidOCR's DetPreProcess."""

    def __init__(self, original):
        self.original = original

    def __call__(self, image):
        resized = self.original.resize(image)
        return None if resized is None else resized[np.newaxis]


class RecPostProcess:
    """CTC decoding from (best index, best probability) instead of the full table."""

    def __init__(self, original):
        self.original = original

    def __call__(self, outputs, return_word_box=False, **kwargs):
        index, probability = outputs
        return self.original.decode(index, probability, return_word_box,
                                    kwargs.get("wh_ratio_list", (1.0,)),
                                    kwargs.get("max_wh_ratio", 1.0), remove_duplicate=True)


class GpuOCR:
    """Callable like a RapidOCR object: image in, result with .txts out."""

    def __init__(self, language="ch"):
        if language != "ch":
            raise RuntimeError(f"the GPU engine is built for the default text model (ch), not {language!r}")
        import onnxruntime
        onnxruntime.preload_dlls()  # CUDA libraries of this environment, needed by TensorRT
        from rapidocr import RapidOCR
        # A CPU-configured RapidOCR supplies the pipeline; its three model calls are replaced.
        self.ocr = RapidOCR(params={"Global.log_level": "critical", "Rec.rec_batch_num": REC_BATCH,
                                    "Cls.cls_batch_num": CLS_BATCH,
                                    "EngineConfig.onnxruntime.intra_op_num_threads": 2})
        blank = np.full((128, 256, 3), 255, np.uint8)
        cv2.putText(blank, "test 123", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self.ocr(blank)  # makes RapidOCR download its models and create its three stages
        ensure_engines()

        det, cls, rec = self.ocr.text_det, self.ocr.text_cls, self.ocr.text_rec
        tuned = engine_path("det", tuned=True)
        self.tuned = os.path.exists(tuned)
        det_run = Runner(tuned if self.tuned else engine_path("det"))
        cls_run = Runner(engine_path("cls"))
        rec_run = Runner(engine_path("rec"))
        preprocess = DetPreProcess(det.get_preprocess())
        det.get_preprocess = lambda: preprocess
        det.session = lambda image: det_run(image)[0]
        cls.session = lambda batch: cls_run(batch)[0]
        rec.session = rec_run
        rec.postprocess_op = RecPostProcess(rec.postprocess_op)

    def __call__(self, image):
        # A long thin image would be scaled past the detection engine's size range, so give
        # it a white margin that keeps its shape within range.
        height, width = image.shape[:2]
        short_side = int(max(height, width) / MAX_ASPECT) + 1
        if min(height, width) < short_side:
            extra = short_side - min(height, width)
            image = cv2.copyMakeBorder(image, 0, extra if height < width else 0,
                                       0, extra if width < height else 0,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return self.ocr(image)
