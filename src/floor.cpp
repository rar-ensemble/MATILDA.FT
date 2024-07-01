#include <iostream>
#include <cmath>

int main()
{
    float lx = 25.0f;
    int nx = 26;
    float x = 24.9;
    float dx = lx/float(nx);
    int g_ind = int(x/dx);
    int g_ind2 = x - (float(g_ind) + 0.5f) * dx;
    std::cout << "dx: " << dx <<" grid id1:"<< g_ind <<" grid id2:"<< g_ind2 <<std::endl;
    if (g_ind2 == nx){
        std::cout << "I am out of bounds for nx " <<nx << std::endl;
    }


    return 0;
}