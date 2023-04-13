workspace(name = "ccv")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
	name = "bazel_skylib",
	remote = "https://github.com/bazelbuild/bazel-skylib.git",
	commit = "528e4241345536c487cca8b11db138104bb3bd68",
	shallow_since = "1601067301 +0200"
)

git_repository(
	name = "build_bazel_rules_cuda",
	remote = "https://github.com/liuliu/rules_cuda.git",
	commit = "be346d4d12883469878edd693097f87723400c5b",
	shallow_since = "1681409802 -0400"
)

http_archive(
	name = "sqlite3",
	sha256 = "87775784f8b22d0d0f1d7811870d39feaa7896319c7c20b849a4181c5a50609b",
	urls = ["https://www.sqlite.org/2022/sqlite-amalgamation-3390200.zip"],
	build_file = "sqlite3.BUILD"
)

load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")
load("@build_bazel_rules_cuda//nccl:nccl_configure.bzl", "nccl_configure")

cuda_configure(name = "local_config_cuda")
nccl_configure(name = "local_config_nccl")

load("//config:ccv.bzl", "ccv_setting")

ccv_setting(
	name = "local_config_ccv",
	have_cblas = True,
	have_libpng = True,
	have_libjpeg = True,
	have_fftw3 = True,
	have_pthread = True,
	have_gsl = True,
	have_cudnn = True,
	have_nccl = True,
	use_openmp = True,
	use_dispatch = True
)
