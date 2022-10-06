config_setting(
    name = "arm_build",
    values = {
        "cpu": "arm"
    }
)

config_setting(
    name = "x86_build",
    values = {
        "cpu": "x86"
    }
)

config_setting(
    name = "have_cuda",
    values = {
        "define": "using_cuda=true"
    }
)

config_setting(
    name = "have_mps",
    values = {
        "define": "enable_mps=true"
    }
)

cc_library(
    name = "cuda_deps",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda"
    ] + %{ccv_setting_cuda_deps},
    linkopts = %{ccv_setting_cuda_linkopts}
)

cc_library(
    name = "config",
    visibility = ["//visibility:public"],
    defines = select({
        ":arm_build": ["HAVE_NEON"],
        ":x86_build": ["HAVE_SSE2"],
        "//conditions:default": []
    }) + select({
        ":have_cuda": ["HAVE_CUDA"],
        "//conditions:default": []
    }) + select({
        ":have_mps": ["HAVE_MPS"],
        "//conditions:default": []
    }) + %{ccv_setting_defines},
    linkopts = ["-lm"] + %{ccv_setting_linkopts}
)
