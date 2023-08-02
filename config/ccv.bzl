load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _ccv_setting_impl(repository_ctx):
    defines = []
    copts = []
    linkopts = []
    cuda_deps = []
    cuda_linkopts = []
    if repository_ctx.attr.have_pthread:
        defines.append("HAVE_PTHREAD")
        linkopts.append("-lpthread")
    
    defines.append("HAVE_ACCELERATE_FRAMEWORK")
    linkopts += ["-framework", "Accelerate"]
    
    if repository_ctx.attr.have_cudnn:
        defines.append("HAVE_CUDNN")
        cuda_linkopts.append("-lcudnn")
    if repository_ctx.attr.have_nccl:
        defines.append("HAVE_NCCL")
        cuda_linkopts.append("-lnccl")
        cuda_deps.append("@local_config_nccl//:nccl")
    if repository_ctx.attr.use_system_cub:
        defines.append("USE_SYSTEM_CUB")
    config = {
        "%{ccv_setting_defines}": str(defines),
        "%{ccv_setting_copts}": str(copts),
        "%{ccv_setting_linkopts}": str(linkopts),
        "%{ccv_setting_cuda_deps}": str(cuda_deps),
        "%{ccv_setting_cuda_linkopts}": str(cuda_linkopts)
    }
    repository_ctx.template("config/BUILD", Label("//config:ccv_setting.BUILD.tpl"), config)
    repository_ctx.template("config/build_defs.bzl", Label("//config:build_defs.bzl.tpl"), config)

# Manual configuration
ccv_setting = repository_rule(
    implementation =_ccv_setting_impl,
    attrs = {
        "have_cblas": attr.bool(),
        "have_libpng": attr.bool(),
        "have_libjpeg": attr.bool(),
        "have_fftw3": attr.bool(),
        "have_pthread": attr.bool(),
        "have_accelerate_framework": attr.bool(),
        "have_gsl": attr.bool(),
        "have_cudnn": attr.bool(),
        "have_nccl": attr.bool(),
        "use_system_cub": attr.bool(),
        "use_openmp": attr.bool(),
        "use_dispatch": attr.bool()
    }
)

# Setup dependencies
def _maybe(repo_rule, name, **kwargs):
    """Executes the given repository rule if it hasn't been executed already.
    Args:
      repo_rule: The repository rule to be executed (e.g., `http_archive`.)
      name: The name of the repository to be defined by the rule.
      **kwargs: Additional arguments passed directly to the repository rule.
    """
    if not native.existing_rule(name):
        repo_rule(name = name, **kwargs)

def ccv_deps():
    _maybe(
        git_repository,
        name = "bazel_skylib",
        remote = "https://github.com/bazelbuild/bazel-skylib.git",
        commit = "528e4241345536c487cca8b11db138104bb3bd68",
        shallow_since = "1601067301 +0200"
    )

    _maybe(
        git_repository,
        name = "build_bazel_rules_cuda",
        remote = "https://github.com/liuliu/rules_cuda.git",
        commit = "be346d4d12883469878edd693097f87723400c5b",
        shallow_since = "1681409802 -0400"
    )
    _maybe(
        http_archive,
        name = "sqlite3",
        sha256 = "87775784f8b22d0d0f1d7811870d39feaa7896319c7c20b849a4181c5a50609b",
        urls = ["https://www.sqlite.org/2022/sqlite-amalgamation-3390200.zip"],
        build_file = "@ccv//:external/sqlite3.BUILD"
    )
