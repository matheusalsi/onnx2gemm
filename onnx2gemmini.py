#!/usr/bin/env python3
"""
onnx2gemmini.py (Robust - com extractor recursivo de pesos QuantizeLinear/Constant/etc.)

Suporta:
 - ONNX float32
 - ONNX já quantizado (INT8 / UINT8 / INT16)
 - ONNX com QuantizeLinear (Brevitas / QONNX)
 - Extrai peso mesmo se passar por nós intermediários (Constant, QuantizeLinear, etc.)
 - Evita double quantization
"""

import onnx
import numpy as np
import os
import argparse
from typing import Optional, Tuple, Dict, Set


# ============================================================
# UTILIDADES
# ============================================================

def to_scalar(x, default=None):
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    try:
        arr = np.array(x).flatten()
        return default if arr.size == 0 else int(arr.tolist()[0])
    except Exception:
        try:
            return int(x)
        except Exception:
            return default


def tensor_to_numpy(tensor):
    # lida com TensorProto em initializers
    if tensor.data_type == onnx.TensorProto.FLOAT:
        dtype = np.float32
    elif tensor.data_type == onnx.TensorProto.INT8:
        dtype = np.int8
    elif tensor.data_type == onnx.TensorProto.UINT8:
        dtype = np.uint8
    elif tensor.data_type == onnx.TensorProto.INT16:
        dtype = np.int16
    elif tensor.data_type == onnx.TensorProto.INT32:
        dtype = np.int32
    elif tensor.data_type == onnx.TensorProto.INT64:
        dtype = np.int64
    else:
        raise NotImplementedError(f"ONNX dtype {tensor.data_type} not supported")

    data = np.frombuffer(tensor.raw_data, dtype=dtype)
    return data.reshape(tensor.dims)


def quantize_tensor_auto(tensor: np.ndarray, precision_bits=8, scale: Optional[float] = None):
    """
    Se tensor já é int8/uint8/int16: usa direto (convertendo para tipo desejado).
    Se tensor é float: quantiza usando scale (se fornecido) ou computa scale por maxabs.
    Retorna (q_tensor, scale_used).
    """
    if tensor.dtype in (np.int8, np.uint8, np.int16):
        q = tensor.astype(np.int8 if precision_bits == 8 else np.int16)
        if scale is None:
            scale = 1.0
        return q, float(scale)

    # float case
    qmax = (2 ** (precision_bits - 1)) - 1
    dtype = np.int8 if precision_bits == 8 else np.int16
    if scale is None:
        maxval = np.max(np.abs(tensor))
        scale = float(maxval / qmax) if maxval != 0 else 1.0
    q = np.round(tensor / scale).astype(dtype)
    return q, float(scale)


def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def compute_conv_output(h_in, w_in, k, stride, pads, dilation):
    h_in, w_in = to_scalar(h_in), to_scalar(w_in)
    if h_in is None or w_in is None:
        raise RuntimeError("Input spatial dims unknown.")
    stride_val, dilation_val = to_scalar(stride, 1), to_scalar(dilation, 1)
    pad_h = pads[0] + pads[2] if pads and len(pads) == 4 else 0
    pad_w = pads[1] + pads[3] if pads and len(pads) == 4 else 0
    kH = kW = to_scalar(k, 1)
    out_h = (h_in + pad_h - dilation_val * (kH - 1) - 1) // stride_val + 1
    out_w = (w_in + pad_w - dilation_val * (kW - 1) - 1) // stride_val + 1
    return int(out_h), int(out_w)


# ============================================================
# RESOLVER RECURSIVO DE PESOS
# ============================================================

def build_index(graph):
    """
    Prepara índices auxiliares para busca rápida:
      - producers_by_output: mapa output_name -> node que produz esse output
      - constants_by_name: Constant node values mapeadas (se houver)
    """
    producers_by_output: Dict[str, onnx.NodeProto] = {}
    constants_by_name: Dict[str, np.ndarray] = {}

    for node in graph.node:
        # index outputs -> node
        for out in node.output:
            producers_by_output[out] = node

        # capture Constant node immediate value
        if node.op_type == "Constant":
            # atributo 'value' é um TensorProto
            val = get_attr(node, "value", None)
            if isinstance(val, onnx.TensorProto):
                try:
                    constants_by_name[node.output[0]] = tensor_to_numpy(val)
                except Exception:
                    pass

    return producers_by_output, constants_by_name


def resolve_input_tensor(name: str, graph: onnx.GraphProto, inits: Dict[str, np.ndarray],
                         producers_by_output: Dict[str, onnx.NodeProto],
                         constants_by_name: Dict[str, np.ndarray],
                         visited: Optional[Set[str]] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Tenta resolver recursivamente um input 'name' até encontrar um initializer (np.ndarray) ou Constant.
    Retorna (tensor_numpy_or_None, scale_or_None)
    """
    if visited is None:
        visited = set()
    if name in visited:
        return None, None
    visited.add(name)

    # Caso direto: é initializer
    if name in inits:
        return inits[name], None

    # Caso Constant resolvido
    if name in constants_by_name:
        return constants_by_name[name], None

    # Se houver produtor, inspeciona o nó
    node = producers_by_output.get(name, None)
    if node is None:
        return None, None

    # Se for QuantizeLinear: inputs [x, scale, zero_point?]
    if node.op_type == "QuantizeLinear":
        # input0: real tensor (pode ser initializer ou Constant ou outro nó)
        real_name = node.input[0]
        scale_name = node.input[1] if len(node.input) > 1 else None

        W, _ = resolve_input_tensor(real_name, graph, inits, producers_by_output, constants_by_name, visited)
        scale_val = None
        if scale_name:
            # scale pode estar em initializer ou constante
            if scale_name in inits:
                arr = inits[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            elif scale_name in constants_by_name:
                arr = constants_by_name[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            else:
                # tenta resolver recursivamente
                s, _ = resolve_input_tensor(scale_name, graph, inits, producers_by_output, constants_by_name, visited)
                if s is not None:
                    arr = np.array(s).reshape(-1)
                    scale_val = float(arr[0]) if arr.size > 0 else None
        return W, scale_val

    # Se for DequantizeLinear: inputs [x, scale, zero_point?] -> x já quantizado (initializer)
    if node.op_type == "DequantizeLinear":
        real_name = node.input[0]
        scale_name = node.input[1] if len(node.input) > 1 else None
        W, _ = resolve_input_tensor(real_name, graph, inits, producers_by_output, constants_by_name, visited)
        scale_val = None
        if scale_name:
            if scale_name in inits:
                arr = inits[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            elif scale_name in constants_by_name:
                arr = constants_by_name[scale_name].reshape(-1)
                scale_val = float(arr[0]) if arr.size > 0 else None
            else:
                s, _ = resolve_input_tensor(scale_name, graph, inits, producers_by_output, constants_by_name, visited)
                if s is not None:
                    arr = np.array(s).reshape(-1)
                    scale_val = float(arr[0]) if arr.size > 0 else None
        return W, scale_val

    # Se for Constant (não capturado antes por alguma razão)
    if node.op_type == "Constant":
        val = get_attr(node, "value", None)
        if isinstance(val, onnx.TensorProto):
            try:
                return tensor_to_numpy(val), None
            except Exception:
                pass

    # Para outros tipos: tenta resolver recursivamente olhando para os inputs do nó
    for inp in node.input:
        if not inp:
            continue
        W, s = resolve_input_tensor(inp, graph, inits, producers_by_output, constants_by_name, visited)
        if W is not None:
            # se achou algum tensor inicial em um dos inputs, devolve ele (sem scale salvo)
            return W, s

    return None, None


def extract_quantized_weight(W_name: str, graph: onnx.GraphProto, inits: Dict[str, np.ndarray]):
    """
    Resolve pesos vindos de QuantizeLinear / Constant / DequantizeLinear etc,
    seguindo cadeias de nós até encontrar o initializer/constant real.
    Retorna (peso_numpy, scale_float)
    """
    producers_by_output, constants_by_name = build_index(graph)

    # Caso direto
    if W_name in inits:
        return inits[W_name], 1.0

    # tenta resolver recursivamente
    W, scale = resolve_input_tensor(W_name, graph, inits, producers_by_output, constants_by_name, visited=set())

    if W is not None:
        # se W for int-typed e scale não foi encontrado, assume scale = 1.0 (fallback)
        return W, (float(scale) if scale is not None else 1.0)

    # Se chegou aqui, não encontrou — produz mensagem útil com sugestões
    available_inits = list(inits.keys())
    # sugestão: procurar nomes "parecidos" entre initializers e W_name
    similar = [n for n in available_inits if n in W_name or W_name in n or n.split('/')[-1] in W_name or W_name.split('/')[-1] in n]
    msg_lines = [
        f"Peso não encontrado: {W_name}",
        f"Initializers disponíveis ({len(available_inits)}): {available_inits[:20]}{'...' if len(available_inits)>20 else ''}",
        f"Initializers com nomes semelhantes: {similar if similar else 'nenhum similar encontrado'}",
        "Sugestão: verifique se o grafo usa QuantizeLinear/Constant e se o initializer do peso tem nome diferente. Aqui estão os nodes cujo output contém a substring do nome requerido:"
    ]
    # lista nodes com outputs que contenham parte do nome
    nodes = []
    for node in graph.node:
        for out in node.output:
            if out and (out in W_name or W_name in out or out.split('/')[-1] in W_name or W_name.split('/')[-1] in out):
                nodes.append((node.op_type, out))
    msg_lines.append(str(nodes if nodes else 'nenhum node similar encontrado'))
    raise KeyError("\n".join(msg_lines))


# ============================================================
# EXPORTADOR
# ============================================================

def export_gemmini(onnx_path, out_dir='out', precision=8, batch_size=4):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    graph = model.graph

    inits = {t.name: tensor_to_numpy(t) for t in graph.initializer}

    node_by_input = {i: [] for node in graph.node for i in node.input if i}
    for node in graph.node:
        for i in node.input:
            if i:
                node_by_input[i].append(node)

    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    _, C, H, W = input_shape if len(input_shape) == 4 else (None, None, None, None)

    basename = os.path.basename(out_dir)
    h_lines, c_lines = [], []

    guard = f"{basename.upper()}_PARAMETERS_H"
    h_lines.append(f"#ifndef {guard}\n#define {guard}\n\n#include <include/gemmini_params.h>\n#include <stdbool.h>\n\n")

    c_lines.extend([
        '#include <stdio.h>\n', '#include <string.h>\n', '#include <stdbool.h>\n',
        '#ifndef BAREMETAL\n', '#include <sys/mman.h>\n', '#endif\n',
        '#include "include/gemmini.h"\n', '#include "include/gemmini_nn.h"\n\n',
        f'#include "{basename}_params.h"\n', '#include "images.h"\n\n',
        'int main (int argc, char * argv[]) {\n',
        '#ifndef BAREMETAL\n',
        '    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { perror("mlockall failed"); exit(1); }\n',
        '#endif\n\n',
        '    gemmini_flush(0);\n\n',
        '    enum tiled_matmul_type_t tiled_matmul_type = WS;\n',
        '    if (argc > 1) {\n',
        '        if (strcmp(argv[1], "cpu") == 0) tiled_matmul_type = CPU;\n',
        '        else if (strcmp(argv[1], "os") == 0) tiled_matmul_type = OS;\n',
        '        else if (strcmp(argv[1], "ws") == 0) tiled_matmul_type = WS;\n',
        '    }\n\n',
        '    uint64_t start, end;\n',
        '    uint64_t conv_cycles = 0, matmul_cycles = 0;\n\n',
        '    // model execution\n'
    ])

    layer_idx = 1
    tensor_buffer, output_dims = {}, {}
    inp_name = graph.input[0].name
    tensor_buffer[inp_name], output_dims[inp_name] = 'images', (C, H, W)
    processed_nodes = set()

    for node in graph.node:
        if node.name in processed_nodes:
            continue

        # =========================
        # CONV
        # =========================
        if node.op_type == 'Conv':
            X_name, W_name = node.input[0], node.input[1]
            B_name = node.input[2] if len(node.input) > 2 else None
            Y_name = node.output[0]

            W_data, scale_real = extract_quantized_weight(W_name, graph, inits)
            B_data = inits.get(B_name, np.zeros(W_data.shape[0], dtype=np.float32))

            out_ch, in_ch_per_group, kH, _ = W_data.shape
            groups = get_attr(node, 'group', 1)
            in_ch = in_ch_per_group * groups

            strides_attr = get_attr(node, 'strides', [1, 1])
            pads_attr = get_attr(node, 'pads', [0, 0, 0, 0])

            _, h_in, w_in = output_dims[X_name]
            out_h, out_w = compute_conv_output(h_in, w_in, kH, strides_attr, pads_attr, [1, 1])
            output_dims[Y_name] = (out_ch, out_h, out_w)

            qW, scaleW = quantize_tensor_auto(W_data, precision, scale_real)
            qB, _ = quantize_tensor_auto(B_data, precision)

            lname = f"conv_{layer_idx}"
            patch_size = in_ch * kH * kH

            w_t = qW.reshape(out_ch, -1).T
            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in w_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"

            n_patches = out_h * out_w * batch_size

            h_lines.extend([
                f"static const elem_t {lname}_w[{patch_size}][{out_ch}] row_align(1) = {w_cstr};\n",
                f"static const acc_t {lname}_b[{out_ch}] row_align_acc(1) = {b_str};\n",
                f"static elem_t {lname}_out[{n_patches}][{out_ch}] row_align(1);\n"
            ])

            tensor_buffer[Y_name] = f"{lname}_out"

            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0

            h_lines.append(
                f"static const struct ConvParams {lname}_params = {{"
                f".batch_size={batch_size}, .in_row_dim={h_in}, .in_col_dim={w_in}, "
                f".kernel_size={kH}, .in_channels={in_ch}, .out_channels={out_ch}, "
                f".stride={to_scalar(strides_attr)}, .padding={to_scalar(pads_attr)}, "
                f".bias=1, .depthwise={1 if groups == in_ch else 0}, "
                f".out_row_dim={out_h}, .out_col_dim={out_w}, "
                f".n_patches={n_patches}, .patch_size={patch_size}, "
                f".pool_size=1, .pool_stride=1, .pool_padding=0, "
                f".out_dim_pooled={out_h}, .output_scale=(1.0 / (1 << {shift})), "
                f".I={n_patches}, .J={out_ch}, .K={patch_size}}};\n\n"
            )

            c_lines.extend([
                f"    // {lname}\n",
                "    start = read_cycles();\n",
                f"    tiled_conv_auto({lname}_params.batch_size, {lname}_params.in_row_dim, "
                f"{lname}_params.in_col_dim, {lname}_params.in_channels, "
                f"{lname}_params.out_channels, {lname}_params.out_row_dim, "
                f"{lname}_params.out_col_dim, {lname}_params.stride, 1, 1, "
                f"{lname}_params.padding, {lname}_params.kernel_size, false, false, false, "
                f"false, false, (elem_t*){tensor_buffer[X_name]}, "
                f"(elem_t*){lname}_w, (acc_t*){lname}_b, (elem_t*){lname}_out, "
                f"NO_ACTIVATION, {lname}_params.output_scale, 1, 1, 0, tiled_matmul_type);\n",
                "    end = read_cycles();\n",
                f"    conv_cycles += end - start;\n"
            ])

            layer_idx += 1
            processed_nodes.add(node.name)

        # =========================
        # GEMM
        # =========================
        elif node.op_type == 'Gemm':
            A_name, W_name, B_name = node.input
            Y_name = node.output[0]

            W_data, scale_real = extract_quantized_weight(W_name, graph, inits)
            B_data = inits[B_name]

            out_features, in_features = W_data.shape

            qW, scaleW = quantize_tensor_auto(W_data, precision, scale_real)
            qB, _ = quantize_tensor_auto(B_data, precision)

            qW_t = qW.T
            lname = f"fc_{layer_idx}"

            rows = ["{" + ",".join(map(str, r.tolist())) + "}" for r in qW_t]
            w_cstr = "{" + ",".join(rows) + "}"
            b_str = "{" + ",".join(map(str, qB.tolist())) + "}"

            h_lines.extend([
                f"static const elem_t {lname}_w[{in_features}][{out_features}] row_align(1) = {w_cstr};\n",
                f"static const acc_t {lname}_b[{out_features}] row_align_acc(1) = {b_str};\n",
                f"static elem_t {lname}_out[{batch_size}][{out_features}] row_align(1);\n"
            ])

            shift = int(np.round(np.log2(1.0 / scaleW))) if scaleW > 0 else 0

            h_lines.append(
                f"static const struct FcParams {lname}_params = {{"
                f".batch_size={batch_size}, .in_features={in_features}, "
                f".out_features={out_features}, .bias=1, "
                f".output_scale=(1.0 / (1 << {shift})), "
                f".I={batch_size}, .J={out_features}, .K={in_features}}};\n\n"
            )

            tensor_buffer[Y_name] = f"{lname}_out"

            c_lines.extend([
                f"    // {lname}\n",
                "    start = read_cycles();\n",
                f"    tiled_matmul_nn_auto({batch_size}, {out_features}, {in_features}, "
                f"(elem_t*){tensor_buffer[A_name]}, (elem_t*){lname}_w, "
                f"(acc_t*){lname}_b, (elem_t*){lname}_out, "
                f"NO_ACTIVATION, 1.0, true, tiled_matmul_type, false, \"{lname}\");\n",
                "    end = read_cycles();\n",
                f"    matmul_cycles += end - start;\n"
            ])

            layer_idx += 1
            processed_nodes.add(node.name)

        else:
            if not node.input or not node.output:
                continue
            input_name = node.input[0]
            output_name = node.output[0]
            if input_name in output_dims:
                output_dims[output_name] = output_dims[input_name]
            if input_name in tensor_buffer:
                tensor_buffer[output_name] = tensor_buffer[input_name]
            processed_nodes.add(node.name)

    c_lines.extend([
        "\n    uint64_t total_cycles = conv_cycles + matmul_cycles;\n",
        '    printf("\\nTotal cycles: %llu (100%%)\\n", total_cycles);\n',
        '    if(total_cycles != 0) {\n',
        '        printf("Matmul cycles: %llu (%d%%)\\n", matmul_cycles, (int)((matmul_cycles * 100) / total_cycles));\n',
        '        printf("Conv cycles: %llu (%d%%)\\n", conv_cycles, (int)((conv_cycles * 100) / total_cycles));\n',
        '    }\n',
        '    printf("PASS\\n");\n\n',
        '    exit(0);\n',
        '}\n'
    ])

    h_lines.append(f"#endif /* {guard} */\n")

    with open(os.path.join(out_dir, f'{basename}_params.h'), 'w') as f:
        f.write(''.join(h_lines))

    with open(os.path.join(out_dir, f'{basename}.c'), 'w') as f:
        f.write(''.join(c_lines))

    print(f"Written {basename}_params.h and {basename}.c to {out_dir}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('onnx', help='ONNX model path')
    p.add_argument('--out', default='out', help='output directory')
    p.add_argument('--precision', type=int, default=8, choices=[8, 16])
    p.add_argument('--batch_size', type=int, default=4)
    args = p.parse_args()

    export_gemmini(args.onnx, out_dir=args.out, precision=args.precision, batch_size=args.batch_size)
