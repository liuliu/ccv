load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _ccv_setting_impl(repository_ctx):
    defines = []
    copts = []
    linkopts = []
    if repository_ctx.attr.have_cblas:
        defines.append("HAVE_CBLAS")
        linkopts += ["-lcblas", "-latlas"]
    if repository_ctx.attr.have_libpng:
        defines.append("HAVE_LIBPNG")
        linkopts.append("-lpng")
    if repository_ctx.attr.have_libjpeg:
        defines.append("HAVE_LIBJPEG")
        linkopts.append("-ljpeg")
    if repository_ctx.attr.have_fftw3:
        defines.append("HAVE_FFTW3")
        linkopts += ["-lfftw3", "-lfftw3f"]
    if repository_ctx.attr.have_pthread:
        defines.append("HAVE_PTHREAD")
        linkopts.append("-lpthread")
    if repository_ctx.attr.have_liblinear:
        defines.append("HAVE_LIBLINEAR")
        linkopts.append("-llinear")
    if repository_ctx.attr.have_tesseract:
        defines.append("HAVE_TESSERACT")
        linkopts.append("-ltesseract")
    if repository_ctx.attr.have_accelerate_framework:
        defines.append("HAVE_ACCELERATE_FRAMEWORK")
        linkopts += ["-framework", "Accelerate"]
    if repository_ctx.attr.have_gsl:
        defines.append("HAVE_GSL")
        linkopts += ["-lgsl", "-lgslcblas"]
    if repository_ctx.attr.have_cudnn:
        defines.append("HAVE_CUDNN")
        linkopts.append("-lcudnn")
    if repository_ctx.attr.have_nccl:
        defines.append("HAVE_NCCL")
        linkopts.append("-lnccl")
    if repository_ctx.attr.have_cub:
        defines.append("HAVE_CUB")
    if repository_ctx.attr.use_openmp:
        defines.append("USE_OPENMP")
        copts.append("-fopenmp")
        linkopts.append("-fopenmp")
    if repository_ctx.attr.use_dispatch:
        defines.append("USE_DISPATCH")
        copts.append("-fblocks") 
        linkopts += ["-ldispatch", "-lBlocksRuntime"]
    config = {
        "%{ccv_setting_defines}": str(defines),
        "%{ccv_setting_copts}": str(copts),
        "%{ccv_setting_linkopts}": str(linkopts)
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
        "have_liblinear": attr.bool(),
        "have_tesseract": attr.bool(),
        "have_accelerate_framework": attr.bool(),
        "have_gsl": attr.bool(),
        "have_cudnn": attr.bool(),
        "have_nccl": attr.bool(),
        "have_cub": attr.bool(),
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
        commit = "836f1b2f564e8952a9b1ae72f66fc9fad8c8e6f1",
        shallow_since = "1599514179 -0600"
    )

    _maybe(
        git_repository,
        name = "build_bazel_rules_cuda",
        remote = "https://github.com/liuliu/rules_cuda.git",
        commit = "816152686a5f9f7f806832c10945b51b4607de29",
        shallow_since = "1599949159 -0400"
    )
