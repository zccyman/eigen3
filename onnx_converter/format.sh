#!/bin/bash

path=$1
selection=$2

case ${selection} in
"python")
        ### format python files
        yapf --in-place --recursive -p --verbose --style .style.yapf ${path}
        ;;
"cpp")
        ### format cpp files
        python run-clang-format.py -r ${path} -i
        ;;
"cmake")
        ### format cmake files
        Files=$(find ${path} -name "CMakeLists.txt" -or -name "*.cmake")

        for file in ${Files}; do
                if [ -f ${file} ]; then
                        # echo $file
                        cmake-format $file -c .cmake-format.py -i
                fi
        done
        ;;
"all")
        ### format python files
        yapf --in-place --recursive -p --verbose --style .style.yapf ${path}
        ### format cpp files
        python run-clang-format.py -r ${path} -i
        ### format cmake files
        Files=$(find ${path} -name "CMakeLists.txt" -or -name "*.cmake")

        for file in ${Files}; do
                if [ -f ${file} ]; then
                        # echo $file
                        cmake-format $file -c .cmake-format.py -i
                fi
        done
        ;;
*)
        echo "${selection} is not exist!!!"
        exit
        ;;
esac
