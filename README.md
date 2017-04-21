# About CMA-VNS

CMA-VNS is an award-winning algorithm. CMA-VNS has won all three tracks in the [CBBOC 2015](//web.mst.edu/~tauritzd/CBBOC/GECCO2015/) competition ([C++ API'15](//github.com/cbboc/cpp/tree/CBBOC-2015)) and all the two tracks in [CBBOC 2016](//web.mst.edu/~tauritzd/CBBOC/GECCO2016/) ([C++ API'16](//github.com/cbboc/cpp/tree/master)).

# How to cite

    Fan Xue and Geoﬀrey Q Shen. 2017. Design of an efficient hyper-heuristic algorithm CMA-VNS for combinatorial black-box optimization problems. In Proceedings of GECCO’17 Companion, Berlin, Germany, July 15-19, 2017, 6 pages. DOI: [http://dx.doi.org/10.1145/3067695.3082054](//dx.doi.org/10.1145/3067695.3082054)
  
# How does it work

As its name, CMA-ES runs first, VNS then takes the baton from CMA-ES in the relay race.

For more details, please refer to the papers.

# When to use CMA-VNS

When your optimization problem is huge, expensive, way too complex.

# Goal of this project

* To develop a competitive entry in CBBOC competitions
* To design an efficient way of solving complex NK-models
* To try to build a bridge between a derivative-free optimization method (CMA-ES) and a local search heuristic (VNS)

# Dependencies

[CBBOC 2015 API](//github.com/cbboc/cpp/tree/CBBOC-2015) or [CBBOC 2016 API](//github.com/cbboc/cpp)

[libcmaes](//github.com/beniz/libcmaes)

[Eigen 3.2+](//eigen.tuxfamily.org)

# Install

* Copy files to src/ folder of CBBOC API
* GCC 4.6+ 

    gcc src/*.cpp -Iinclude -Ipath_to_libcmaes -Ipath_to_Eigen -std=c++0x -lstdc++ -O3

# How to contribute

# License

LGPL-3.0


 
