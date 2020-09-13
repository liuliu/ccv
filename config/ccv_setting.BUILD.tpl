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

cc_library(
    name = "config",
    srcs = [],
    hdrs = [],
    visibility = ["//visibility:public"],
    defines = select({
        ":arm_build": ["HAVE_NEON"],
        ":x86_build": ["HAVE_SSE2"],
        "//conditions:default": []
    }) + select({
        ":have_cuda": ["HAVE_CUDA"],
        "//conditions:default": []
    })+ %{ccv_setting_defines},
    linkopts = ["-lm"] + %{ccv_setting_linkopts}
)
