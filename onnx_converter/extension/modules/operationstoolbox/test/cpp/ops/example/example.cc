#include <iostream>

#include "add.h"

int main(int argc, char **argv)
{
    int a = 2;
    int b = 1;

    int c = add(a, b);

    std::cout << "c = " << c << std::endl;

    return 0;
}