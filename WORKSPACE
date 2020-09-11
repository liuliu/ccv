load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
	name = "bazel_skylib",
	remote = "https://github.com/bazelbuild/bazel-skylib.git",
	commit = "836f1b2f564e8952a9b1ae72f66fc9fad8c8e6f1",
	shallow_since = "1599514179 -0600"
)

git_repository(
	name = "build_bazel_rules_cuda",
	remote = "https://github.com/liuliu/rules_cuda.git",
	commit = "6698cfba363fb69e5a5c07cc7db1c363f9171478",
	shallow_since = "1599857489 -0400"
)

load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")
