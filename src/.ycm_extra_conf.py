import os
import ycm_core

flags = []

compilation_database_folder = '../bin/'
if os.path.exists(compilation_database_folder):
    database = ycm_core.CompilationDatabase(compilation_database_folder)
else:
    database = None

SOURCE_EXTENSIONS = ['.cpp']
HEADER_EXTENSIONS = ['.hpp', '.h']


def DirectoryOfThisScript():
    return os.path.dirname(os.path.abspath(__file__))


def IsHeaderFile(filename):
    extension = os.path.splitext(filename)[1]
    return extension in HEADER_EXTENSIONS


def GetCompilationInfoForFile(filename):
    if IsHeaderFile(filename):
        basename = os.path.splitext(filename)[0]
        for extension in SOURCE_EXTENSIONS:
            replacement_file = basename + extension
            if os.path.exists(replacement_file):
                compilation_info = database.GetCompilationInfoForFile(replacement_file)
                if compilation_info.compiler_flags_:
                    return compilation_info
        return None
    return database.GetCompilationInfoForFile(filename)

def FlagsForFile(filename, **kwargs):
    if database:
        compilation_info = GetCompilationInfoForFile(filename)
        if not compilation_info:
            return None

        final_flags = compilation_info.compiler_flags_
    else:
        relative_to = DirectoryOfThisScript()
        final_flags = MakeRelativePathsInFlagsAbsolute(flags, relative_to)

    crutch = ['-isystem', '/usr/include/c++/6.1.1',
              '-isystem', '/usr/lib/gcc/i686-redhat-linux/6.1.1/../../../../include/c++/6.1.1',
              '-isystem', '/usr/lib/gcc/i686-redhat-linux/6.1.1/../../../../include/c++/6.1.1/i686-redhat-linux',
              '-isystem', '/usr/lib/gcc/i686-redhat-linux/6.1.1/../../../../include/c++/6.1.1/backward',
              '-isystem', '/usr/lib/gcc/i686-redhat-linux/6.1.1/include',
              '-isystem', '/usr/local/include',
              '-isystem', '/usr/include']

    final_flags.extend(crutch)
    return {'flags': final_flags,
            'do_cache' : True}
