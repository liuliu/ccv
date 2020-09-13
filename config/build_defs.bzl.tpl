
def ccv_default_copts():
    return select({
        "//config:arm_build": ["-mfpu=neon", "-mfloat-abi=hard"],
        "//config:x86_build": ["-msse2"],
        "//conditions:default": []
    }) + %{ccv_setting_copts}
