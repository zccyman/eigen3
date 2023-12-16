# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/3/28 9:58
# @File     : setup.py

# import collections
import collections
import distutils
import os
import glob
import copy
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import warnings
import setuptools
from tkinter import _flatten
from pkg_resources import packaging
from typing import List, Optional, Union

from setuptools import find_packages
# from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as _build_py
from pkg_resources import parse_version, get_distribution, DistributionNotFound

import multiprocessing
from setuptools import setup, find_packages
# from distutils.core import setup
from distutils.command import clean
# from distutils.extension import Extension
from distutils.dir_util import remove_tree
# from distutils.errors import DistutilsSetupError
# from distutils.sysconfig import customize_compiler, get_python_version
# from distutils.sysconfig import get_config_h_filename
# from distutils.dep_util import newer_group
# from distutils.extension import Extension
# from distutils.util import get_platform
from distutils import log

from Cython.Build import cythonize
import torch
import shutil

# copy_extensions_to_source
try:
    from Cython.Distutils.build_ext import new_build_ext as build_ext
except:
    from Cython.Distutils import build_ext

import zipfile
import git


# copy source file to site-packages
packages = 'onnx_converter'
# if not os.path.exists(packages):
    # shutil.copytree('onnx-converter', packages)
    # shutil.rmtree('onnx-converter')
py_packages = set(find_packages()) | {packages}
# packages = 'onnx_converter'
# if os.path.exists('onnx-converter'):
#    os.rename('onnx-converter', packages)

def read():
    with open(f"{packages}/benchmark/requirement.txt", "r") as f:
        return f.readlines()
    
repo = git.Repo(packages)
branch_name = "master"
branch = repo.branches[branch_name]
commit_id = str(branch.commit.hexsha).strip()
commit_id_path = f"{packages}/utest/data"
if not os.path.exists(commit_id_path):
    os.makedirs(commit_id_path)
with open(os.path.join(commit_id_path, "commit_id.txt"), "w") as f:
    f.write(":".join([branch_name, commit_id]))

# Notice2: exclude files.
EXCLUDE_FILES = [
    # 'your package/ignore python files.'
    # 'package/file1.py'
    f'{packages}/.cmake-format.py',
    f'{packages}/format.sh',
    f'{packages}/run-clang-format.py',
    f'{packages}/test.py'
]
EXCLUDE_PATHS = [
    # 'your package/ignore python path.'
    # 'package/'
    'trained_mdoels',
    'benchmark',
    'eval',
    'tests',
]

IS_WINDOWS = sys.platform == 'win32'
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'
EXTRA_LIB_PATH = os.path.join(packages, 'lib')
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)
ABI_INCOMPATIBILITY_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({}) may be ABI-incompatible!
Please use a compiler that is ABI-compatible with GCC 5.0 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 5 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''
WRONG_COMPILER_WARNING = '''

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler ({user_compiler}) is not compatible with the compiler was built
with for this platform.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!
'''


def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home:
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'")
    return cuda_home


CUDA_HOME = _find_cuda_home()
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\w+\+\w+')

COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/EHsc']

MSVC_IGNORE_CUDAFE_WARNINGS = [
    # 'base_class_has_different_dll_interface',
    # 'field_without_dll_interface',
    # 'dll_interface_conflict_none_assumed',
    # 'dll_interface_conflict_dllexport_assumed'
]

COMMON_NVCC_FLAGS = [
    # '-D__CUDA_NO_HALF_OPERATORS__',
    # '-D__CUDA_NO_HALF_CONVERSIONS__',
    # '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    # '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

PLAT_TO_VCVARS = {
    'win32': 'x86',
    'win-amd64': 'x86_amd64',
}


def get_ext_paths(root_dir, exclude_files, exclude_paths):
    """get filepaths for compilation"""
    paths = []
    # exclude_paths_ = [os.path.join(root_dir, ex_paths) for ex_paths in exclude_paths]
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            # print('################################################', exclude_paths_, os.path.dirname(file_path))
            is_skip = False
            for ex_path in exclude_paths:
                if ex_path in file_path:
                    is_skip = True
                    break
            is_skip = is_skip or file_path in exclude_files
            if is_skip:
                continue

            paths.append(file_path)
    return paths


def cleanup_build_outputs(root_dir, packages):
    try:
        remove_tree(os.path.join(root_dir, packages + '.egg-info'))
    except:
        print('remove {}.egg-info falied!'.format(packages))

    try:
        remove_tree(os.path.join(root_dir, 'build'))
    except:
        print('remove build falied!')
    '''
    try:
        target_file = os.listdir(os.path.join(root_dir, 'dist'))[0]
        shutil.copy(os.path.join(root_dir, 'dist', target_file), root_dir)
        print(os.path.join(root_dir, 'dist', target_file))
        remove_tree(os.path.join(root_dir, 'dist'))
    except:
       print('remove dist falied!')
    '''

    if os.path.exists(packages):
        shutil.rmtree(packages)


def copy_files(root_dir, target_dir, files, func_name):
    for f in files:
        diff = os.path.commonpath([root_dir, f])
        diff = f.replace(diff, '')
        dest = target_dir + diff
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        print('diff: %s, cp %s -> %s' % (diff, f, dest))
        func_name(f, dest)


def search_files_recursively(root_dir, pattern):
    '''Searches files inside subdirectories, including root directory
    '''
    scan_root = os.path.join(root_dir, pattern)
    files = glob.glob(scan_root)
    scan_root = os.path.join(root_dir, '**', pattern)
    files += glob.glob(scan_root)
    return files


def include_paths(cuda: bool = False) -> List[str]:
    lib_include = os.path.join(packages, 'include')
    paths = [
        lib_include,
        os.path.join(lib_include, '', 'include'),
    ]
    if cuda:
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    return paths


def library_paths(cuda: bool = False) -> List[str]:
    paths = [EXTRA_LIB_PATH]

    if cuda:
        if IS_WINDOWS:
            lib_dir = 'lib/x64'
        else:
            lib_dir = 'lib64'
            if (not os.path.exists(_join_cuda_home(lib_dir)) and
                    os.path.exists(_join_cuda_home('lib'))):
                # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                # Note that it's also possible both don't exist (see
                # _find_cuda_home) - in that case we stay with 'lib64'.
                lib_dir = 'lib'

        paths.append(_join_cuda_home(lib_dir))
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, lib_dir))
    return paths


def _join_cuda_home(*paths) -> str:
    r'''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def CppExtension(name, sources, *args, **kwargs):
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    '''
    setup(
        name='cuda_extension',
        ext_modules=[
            CUDAExtension(
                    name='cuda_extension',
                    sources=['extension.cpp', 'extension_kernel.cu'],
                    extra_compile_args={'cxx': ['-g'],
                                        'nvcc': ['-O2']})
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('lib')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    return setuptools.Extension(name, sources, *args, **kwargs)


class clean(clean.clean):
    def run(self):
        delete_dirs = ['onnx_converter.egg-info', 'build']
        for wildcard in delete_dirs:
            for filename in glob.glob(wildcard):
                try:
                    os.remove(filename)
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


def is_ninja_available():
    r'''
    Returns ``True`` if the `ninja <https://ninja-build.org/>`_ build system is
    available on the system, ``False`` otherwise.
    '''
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True


def _accepted_compilers_for_platform() -> List[str]:
    # gnu-c++ and gnu-cc are the conda gcc compilers
    return ['clang++', 'clang'] if sys.platform.startswith('darwin') else ['g++', 'gcc', 'gnu-c++', 'gnu-cc']


def check_compiler_ok_for_platform(compiler: str) -> bool:
    if IS_WINDOWS:
        return True
    which = subprocess.check_output(['which', compiler], stderr=subprocess.STDOUT)
    # Use os.path.realpath to resolve any symlinks, in particular from 'c++' to e.g. 'g++'.
    compiler_path = os.path.realpath(which.decode(*SUBPROCESS_DECODE_ARGS).strip())
    # Check the compiler name
    if any(name in compiler_path for name in _accepted_compilers_for_platform()):
        return True
    # If ccache is used the compiler path is /usr/bin/ccache. Check by -v flag.
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT).decode(*SUBPROCESS_DECODE_ARGS)
    if sys.platform.startswith('linux'):
        # Check for 'gcc' or 'g++'
        pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            return False
        compiler_path = os.path.realpath(results[0].strip())
        return any(name in compiler_path for name in _accepted_compilers_for_platform())
    if sys.platform.startswith('darwin'):
        # Check for 'clang' or 'clang++'
        return version_string.startswith("Apple clang")
    return False


def _is_binary_build() -> bool:
    return True


def _get_cuda_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    r'''
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    '''
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if 'arch' in flag:
                return []

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere', '8.0;8.6+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # If not given, determine what's best for the GPU / CUDA version that can be found
    if not _arch_list:
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int(arch.split('_')[1])
                            for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
            # Capability of the device may be higher than what's supported by the user's
            # NVCC, causing compilation error. User's NVCC is expected to match the one
            # used to build pytorch, so we use the maximum supported capability of pytorch
            # to clamp the capability.
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
        arch_list[-1] += '+PTX'
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        _arch_list = _arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archval in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archval)

        arch_list = _arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            num = arch[0] + arch[2]
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    return sorted(list(set(flags)))


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    return os.path.splitext(path)[1] in valid_ext


def verify_ninja_availability():
    r'''
    Raises ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not
    available on the system, does nothing otherwise.
    '''
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions")


def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda:
        nvcc = _join_cuda_home('bin', 'nvcc')
        config.append(f'nvcc = {nvcc}')

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile was added in CUDA 10.2.
        # Compilation will work on earlier CUDA versions but header file
        # dependencies are not correctly computed.
        required_cuda_version = packaging.version.parse('10.2')
        has_cuda_version = torch.version.cuda is not None
        if has_cuda_version and packaging.version.parse(torch.version.cuda) >= required_cuda_version:
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            if IS_WINDOWS:
                nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [link_rule, build, link, default]
    with open(path, 'w') as build_file:
        for block in blocks:
            lines = '\n'.join(block)
            build_file.write(f'{lines}\n\n')


def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...')
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... '
              '(overridable by setting the environment variable MAX_JOBS=N)')
    return None


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]

        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detachs __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        # it is valid.
        stdout_fileno = 1
        subprocess.run(
            command,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env)
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `ouput`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _write_ninja_file_and_compile_objects(
        sources: List[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool]) -> None:
    verify_ninja_availability()
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    check_compiler_abi_compatibility(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...')
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda)
    if verbose:
        print('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix='Error compiling objects for extension')


def _nt_quote_args(args: Optional[List[str]]) -> List[str]:
    """Quote command-line arguments for DOS/Windows conventions.

    Just wraps every argument which contains blanks in double quotes, and
    returns a new argument list.
    """
    # Cover None-type
    if not args:
        return []
    return [f'"{arg}"' if ' ' in arg else arg for arg in args]


def check_compiler_abi_compatibility(compiler) -> bool:
    if not _is_binary_build():
        return True

    # First check if the compiler is one of the expected ones for the particular platform.
    if not check_compiler_ok_for_platform(compiler):
        warnings.warn(WRONG_COMPILER_WARNING.format(
            user_compiler=compiler,
            platform=sys.platform))
        return False

    if sys.platform.startswith('darwin'):
        # There is no particular minimum version we need for clang, so we're good here.
        return True
    try:
        if sys.platform.startswith('linux'):
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
            version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split('.')
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
            match = re.search(r'(\d+)\.(\d+)\.(\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
            version = (0, 0, 0) if match is None else match.groups()
    except Exception:
        _, error, _ = sys.exc_info()
        warnings.warn(f'Error checking compiler version for {compiler}: {error}')
        return False

    if tuple(map(int, version)) >= minimum_required_version:
        return True

    compiler = f'{compiler} {".".join(version)}'
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))

    return False


class TimesIntelliCPPCUDABuildExt(build_ext, object):

    @classmethod
    def with_options(cls, **options):
        r'''
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        '''

        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        self.complier_setting = {"COMPILER_TYPE": '_gcc',
                                 "STDLIB": '_libstdcpp',
                                 "BUILD_ABI": '_cxxabi1011',
                                 '_GLIBCXX_USE_CXX11_ABI': False}

        super(TimesIntelliCPPCUDABuildExt, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '{}. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def _check_cuda_version(self):
        CUDA_MISMATCH_MESSAGE = '''
        The detected CUDA version ({0}) mismatches the version that was used to compile
        nvcc ({1}). Please make sure to use the same CUDA versions.
        '''
        CUDA_MISMATCH_WARN = '''The detected CUDA version ({0}) has a minor version mismatch with the 
        version that was used to compile nvcc ({1}). Most likely this shouldn't be a problem.'''

        if CUDA_HOME:
            nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
            cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
            cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
            if cuda_version is not None:
                cuda_str_version = cuda_version.group(1)
                cuda_ver = packaging.version.parse(cuda_str_version)
                timeintelli_cuda_version = packaging.version.parse(torch.version.cuda)
                if cuda_ver != timeintelli_cuda_version:
                    # major/minor attributes are only available in setuptools>=49.6.0
                    if getattr(cuda_ver, "major", float("nan")) != getattr(timeintelli_cuda_version, "major",
                                                                           float("nan")):
                        raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
                    warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))

        else:
            raise RuntimeError('CUDA_NOT_FOUND_MESSAGE')

    def build_extensions(self):
        self._check_abi()

        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        if cuda_ext:
            self._check_cuda_version()

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict. Otherwise, default compile flags do
            # not get passed. Necessary when only one of 'cxx' and 'nvcc' is
            # passed to extra_compile_args in CUDAExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, '-DTIMRSINTELLI_INCLUDE_EXTENSION_H')
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = self.complier_setting[name]
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_TimesIntelli_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # Register .cu, .cuh and .hip as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std14_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++14'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (COMMON_NVCC_FLAGS +
                      ['--compiler-options', "'-fPIC'"] +
                      cflags + _get_cuda_arch_flags(cflags))

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if (
                    _ccbin is not None
                    and not any([flag.startswith('-ccbin') or flag.startswith('--compiler-bindir') for flag in cflags])
            ):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = _join_cuda_home('bin', 'nvcc')
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                append_std14_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std14_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std14_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return (COMMON_NVCC_FLAGS +
                    cflags + _get_cuda_arch_flags(cflags))

        def win_wrap_single_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        cflags = win_cuda_flags(cflags) + ['--use-local-env']
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                            cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(sources,
                                   output_dir=None,
                                   macros=None,
                                   include_dirs=None,
                                   debug=0,
                                   extra_preargs=None,
                                   extra_postargs=None,
                                   depends=None):

            if not self.compiler.initialized:
                self.compiler.initialize()
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any,
            # For ninja build, the build location is not local, the build happens
            # in a in script created build folder, relative path lost their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = \
                self.compiler._setup_compile(output_dir, macros,
                                             include_dirs, sources,
                                             depends, extra_postargs)
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            common_cflags.extend(COMMON_MSVC_FLAGS)
            cflags = cflags + common_cflags + pp_opts
            with_cuda = any(map(_is_cuda_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std14_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['--use-local-env']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                    cuda_cflags.append('-Xcudafe')
                    cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = win_cuda_flags(cuda_post_cflags)

            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda)

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super(TimesIntelliCPPCUDABuildExt, self).get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def _check_abi(self):
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        elif IS_WINDOWS:
            compiler = os.environ.get('CXX', 'cl')
        else:
            compiler = os.environ.get('CXX', 'c++')
        check_compiler_abi_compatibility(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = ('It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
                   'This may lead to multiple activations of the VC env.'
                   'Please set `DISTUTILS_USE_SDK=1` and try again.')
            raise UserWarning(msg)

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    # framework complie tools special parameter
    def _define_TimesIntelli_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTIMESINTELLI_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what compile was compiled with
        self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + \
                               str(int(self.complier_setting['_GLIBCXX_USE_CXX11_ABI'])))

    def swig_sources(self, sources, extension):
        """Walk the list of source files in 'sources', looking for SWIG
        interface (.i) files.  Run SWIG on all that are found, and
        return a modified 'sources' list with SWIG source files replaced
        by the generated C (or C++) files.
        """
        new_sources = []
        swig_sources = []
        swig_targets = {}

        # XXX this drops generated C/C++ files into the source tree, which
        # is fine for developers who want to distribute the generated
        # source -- but there should be an option to put SWIG output in
        # the temp dir.

        if self.swig_cpp:
            log.warn("--swig-cpp is deprecated - use --swig-opts=-c++")

        if self.swig_cpp or ('-c++' in self.swig_opts) or \
                ('-c++' in extension.swig_opts):
            target_ext = '.cpp'
        else:
            target_ext = '.c'

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == ".i":  # SWIG interface file
                new_sources.append(base + '_wrap' + target_ext)
                swig_sources.append(source)
                swig_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not swig_sources:
            return new_sources

        swig = self.swig or self.find_swig()
        swig_cmd = [swig, "-python"]
        swig_cmd.extend(self.swig_opts)
        if self.swig_cpp:
            swig_cmd.append("-c++")

        # Do not override commandline arguments
        if not self.swig_opts:
            for o in extension.swig_opts:
                swig_cmd.append(o)

        for source in swig_sources:
            target = swig_targets[source]
            log.info("swigging %s to %s", source, target)
            self.spawn(swig_cmd + ["-o", target, source])

        return new_sources


class TimesIntelliBuildExt(build_ext, object):

    def __init__(self, *args, **kwargs) -> None:
        super(TimesIntelliBuildExt, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = ('Attempted to use ninja as the BuildExtension backend but '
                   '{}. Falling back to using the slow distutils backend.')
            if not is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def run(self):
        self.debug = False
        super(TimesIntelliBuildExt, self).run()

        root_dir = os.getcwd()
        build_dir = os.path.join(root_dir, self.build_lib)
        target_dir = build_dir if not self.inplace else root_dir

        # Identify all init.py
        scan_root = os.path.join(root_dir, packages)
        # pattern = '__init__.py'
        # print('Searching for __init__.py in ', scan_root)
        # __init__files = search_files_recursively(scan_root, '__init__.py')
        # copy_files(root_dir, target_dir, __init__files, shutil.copy)
        cython_files = search_files_recursively(scan_root, '*.c')
        copy_files(root_dir, target_dir, cython_files, shutil.move)
        # cleanup_build_outputs(root_dir)

    # build_ext source files tar
    def copy_extensions_to_source(self):
        super(TimesIntelliBuildExt, self).copy_extensions_to_source()

    def swig_sources(self, sources, extension):
        """Walk the list of source files in 'sources', looking for SWIG
        interface (.i) files.  Run SWIG on all that are found, and
        return a modified 'sources' list with SWIG source files replaced
        by the generated C (or C++) files.
        """
        new_sources = []
        swig_sources = []
        swig_targets = {}

        # XXX this drops generated C/C++ files into the source tree, which
        # is fine for developers who want to distribute the generated
        # source -- but there should be an option to put SWIG output in
        # the temp dir.

        if self.swig_cpp:
            log.warn("--swig-cpp is deprecated - use --swig-opts=-c++")

        if self.swig_cpp or ('-c++' in self.swig_opts) or \
                ('-c++' in extension.swig_opts):
            target_ext = '.cpp'
        else:
            target_ext = '.c'

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == ".i":  # SWIG interface file
                new_sources.append(base + '_wrap' + target_ext)
                swig_sources.append(source)
                swig_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not swig_sources:
            return new_sources

        swig = self.swig or self.find_swig()
        swig_cmd = [swig, "-python"]
        swig_cmd.extend(self.swig_opts)
        if self.swig_cpp:
            swig_cmd.append("-c++")

        # Do not override commandline arguments
        if not self.swig_opts:
            for o in extension.swig_opts:
                swig_cmd.append(o)

        for source in swig_sources:
            target = swig_targets[source]
            log.info("swigging %s to %s", source, target)
            self.spawn(swig_cmd + ["-o", target, source])

        return new_sources


# noinspection PyPep8Naming
# source file build in site-packages directions
class TimesIntelliBuildSourceFiles(_build_py):

    def find_package_modules(self, package, package_dir):
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for (pkg, mod, filepath) in modules:
            if os.path.exists(filepath.replace('.py', ext_suffix)):
                continue
            filtered_modules.append((pkg, mod, filepath,))
        cleanup_build_outputs()
        return filtered_modules


def find_library(name, vision_include):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    build_prefix = os.environ.get("BUILD_PREFIX", None)
    is_conda_build = build_prefix is not None

    library_found = False
    conda_installed = False
    lib_folder = None
    include_folder = None
    library_header = f"{name}.h"

    # Lookup in TORCHVISION_INCLUDE or in the package file
    # third party library paths
    package_path = [os.path.join(this_dir, "")]
    for folder in vision_include + package_path:
        candidate_path = os.path.join(folder, library_header)
        library_found = os.path.exists(candidate_path)
        if library_found:
            break

    if not library_found:
        print(f"Running build on conda-build: {is_conda_build}")
        if is_conda_build:
            # Add conda headers/libraries
            if os.name == "nt":
                build_prefix = os.path.join(build_prefix, "Library")
            include_folder = os.path.join(build_prefix, "include")
            lib_folder = os.path.join(build_prefix, "lib")
            library_header_path = os.path.join(include_folder, library_header)
            library_found = os.path.isfile(library_header_path)
            conda_installed = library_found
        else:
            # Check if using Anaconda to produce wheels
            conda = distutils.spawn.find_executable("conda")
            is_conda = conda is not None
            print(f"Running build on conda: {is_conda}")
            if is_conda:
                python_executable = sys.executable
                py_folder = os.path.dirname(python_executable)
                if os.name == "nt":
                    env_path = os.path.join(py_folder, "Library")
                else:
                    env_path = os.path.dirname(py_folder)
                lib_folder = os.path.join(env_path, "lib")
                include_folder = os.path.join(env_path, "include")
                library_header_path = os.path.join(include_folder, library_header)
                library_found = os.path.isfile(library_header_path)
                conda_installed = library_found

        if not library_found:
            if sys.platform == "linux":
                library_found = os.path.exists(f"/usr/include/{library_header}")
                library_found = library_found or os.path.exists(f"/usr/local/include/{library_header}")

    return library_found, conda_installed, include_folder, lib_folder


extension_modules = cythonize(
    get_ext_paths(packages, EXCLUDE_FILES, EXCLUDE_PATHS),  #
    nthreads=multiprocessing.cpu_count(),
    compiler_directives={
        'language_level': 3.7,
        'always_allow_keywords': True
    },
    build_dir="build",
)


def get_build_extensions(root_dir):
    flatten_list = lambda inputs: list(_flatten(inputs))
    cpu_prefixs = ['*.c', '*.cpp', '*.cxx']
    cuda_prefixs = ['*.c', '*.cpp', '*.cxx', '*.cu', '*.cuh']
    search_source = lambda root_dir, path, prefixs: [glob.glob(os.path.join(root_dir, path, prefix)) for prefix in
                                                     prefixs]
    cpu_root_dir = os.path.join(root_dir, 'cpu')
    main_file = search_source(root_dir, '', cpu_prefixs) + flatten_list(search_source(cpu_root_dir, '', cpu_prefixs))

    source_cpu = [search_source(cpu_root_dir, sub_path, cpu_prefixs) for sub_path in os.listdir(cpu_root_dir)]

    cuda_root_dir = os.path.join(root_dir, 'cuda')
    source_cuda = search_source(root_dir, '', ['*.cu', '*.cuh']) + flatten_list(
        search_source(cuda_root_dir, '', cuda_prefixs))

    sources = main_file + flatten_list(source_cpu)

    extension = CppExtension

    define_macros = []

    extra_compile_args = {"cxx": []}

    if torch.cuda.is_available() and len(source_cuda) > 1:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")

        extra_compile_args["nvcc"] = nvcc_flags

    if sys.platform == "win32":
        define_macros += [("USE_PYTHON", None)]
        extra_compile_args["cxx"].append("/MP")

    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compile in debug mode")
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [f for f in nvcc_flags if not ("-O" in f or "-g" in f)]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")

    include_dirs = [root_dir]
    ext_modules = [
        extension(
            "onnx_converter.ops",
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    '''
    # example for load third patry library
    ext_modules.append(
        CppExtension(
            "onnx_converter.ops",
            src_path,
            include_dirs=[
                include_dir,
                root_dir,
            ],
            library_dirs=library_dirs,
            libraries=[
                "cudart",
            ],
            extra_compile_args=["-std=c++14"] if os.name != "nt" else ["/std:c++14", "/MP"],
            extra_link_args=["-std=c++14" if os.name != "nt" else "/std:c++14"],
        )
    )
    '''

    return ext_modules

def zipDir(source_dir, zip_name):
    if os.path.isdir(source_dir):
        zip=zipfile.ZipFile(zip_name,'w')
        for abs_path,dirnames,filenames in os.walk(source_dir):
            rel_path=abs_path.replace(source_dir,'') 
            for filename in filenames:
                zip.write(os.path.join(abs_path,filename),os.path.join(rel_path,filename))
        zip.close()
    else:
        with zipfile.ZipFile(zip_name, 'w') as zip:

            filename=os.path.split(source_dir)[-1]
            zip.write(source_dir,filename)

# extension_modules.append(get_build_extensions(packages))

setup(
    name=packages,  #
    version='2.0.4',
    description='This package is TimesIntelli TECH Develop',
    author='TimesIntelli Develop Team',
    author_email='candy.yuan@timesintelli.com;shengyuan.shen@timesintelli.com;henson.zhang@timesintelli.com;nan.qin@timesintelli.com;',
    url='http://www.timesintelli.com/',
    ext_modules=extension_modules,
    cmdclass={
        # 'build_py': TimesIntelliBuildSourceFiles,
        'build_ext': TimesIntelliBuildExt,
        #'build_ext': TimesIntelliCPPCUDABuildExt.with_options(no_python_abi_suffix=True),
        'clean': clean
    },
    options={
        'build_ext':
            {'parallel': multiprocessing.cpu_count()},
    },
    include_package_data=True,
    packages=[
        f'{packages}/perf_data', 
        f'{packages}/utest/data', 
        f'{packages}/extension/libs',
    ],
    install_requires=read(),
    # cleanup_build_outputs=cleanup_build_outputs(),
)



cleanup_build_outputs('./', packages)

# copy static files
# static_files_folder = 'onnx-converter/perf_data/'
# target_path = 'onnx_converter/perf_data'
# whl_file_name = [x for x in os.listdir('dist/') if x.endswith('.whl')][0]
# whl_path = os.path.join("dist", whl_file_name)
# zfile = zipfile.ZipFile(whl_path,'r')
# #print(zfile.namelist())
# unzip_path = 'dist/tmp'
# if os.path.exists(unzip_path):
#     print("delete %s"%(unzip_path))
#     shutil.rmtree(unzip_path)
# else:
#     os.makedirs(unzip_path)
# for file in zfile.namelist():
#     zfile.extract(file, unzip_path)
# #os.rename(whl_path, whl_path.replace(".whl", ".whl_copy"))
# target_path = os.path.join(unzip_path, target_path)
# shutil.copytree(static_files_folder, target_path)
# # remove non-encrypted csv files
# for file in os.listdir(target_path):
#     for chip_model in os.listdir(target_path):
#         subdir = os.path.join(target_path, chip_model)
#         for file in os.listdir(subdir):
#             if file.endswith('.csv'):
#                 os.remove(os.path.join(subdir, file))

# zipDir(unzip_path, whl_path)
# shutil.rmtree(unzip_path)





