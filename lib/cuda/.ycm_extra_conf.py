import os
import ycm_core
from clang_helpers import PrepareClangFlags

flags = [
	'-ffast-math',
	'-Wall',
	'-msse2',
	'-D HAVE_SSE2',
	'-D HAVE_LIBJPEG',
	'-D HAVE_LIBPNG',
	'-D HAVE_GSL',
	'-D HAVE_FFTW3',
	'-D HAVE_LIBLINEAR',
	'-D HAVE_CBLAS',
	'-D HAVE_AVCODEC',
	'-D HAVE_AVFORMAT',
	'-D HAVE_SWSCALE',
	'-I',
	'..'
]

def DirectoryOfThisScript():
	return os.path.dirname(os.path.abspath(__file__))

def MakeRelativePathsInFlagsAbsolute(flags, working_directory):
	if not working_directory:
		return flags
	new_flags = []
	make_next_absolute = False
	path_flags = ['-isystem', '-I', '-iquote', '--sysroot=']
	for flag in flags:
		new_flag = flag

		if make_next_absolute:
			make_next_absolute = False
			if not flag.startswith('/'):
				new_flag = os.path.join(working_directory, flag)

		for path_flag in path_flags:
			if flag == path_flag:
				make_next_absolute = True
				break

			if flag.startswith(path_flag):
				path = flag[len(path_flag):]
				new_flag = path_flag + os.path.join(working_directory, path)
				break

		if new_flag:
			new_flags.append(new_flag)
	return new_flags

def FlagsForFile(filename):
	relative_to = DirectoryOfThisScript()
	final_flags = MakeRelativePathsInFlagsAbsolute(flags, relative_to)
	return {
		'flags' : final_flags,
		'do_cache' : True
	}
